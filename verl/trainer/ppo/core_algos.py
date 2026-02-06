# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_grpo_extended_advantage(token_level_rewards: torch.Tensor,
                                    values: torch.Tensor,
                                    eos_mask: torch.Tensor,
                                    index: torch.Tensor,
                                    gamma: float = 1.0,
                                    epsilon: float = 1e-6):
    """
    Compute advantage for GRPO with dense rewards (Short-term + Long-term).
    Short-term reward: values (predicted score)
    Long-term reward: Discounted sum of future short-term rewards.
    Total: Short-term + Long-term (which is the Return).
    
    Args:
        token_level_rewards: Outcome rewards (Sparse).
        values: Short-term predicted scores (Dense).
        eos_mask: Mask for valid tokens.
        index: Group index.
        gamma: Discount factor.
    """
    
    # 1. Combine rewards: r_t = values[t] + outcome[t]
    # Note: outcome[t] is usually 0 except at EOS.
    # We assume 'values' contains the "Short-term reward" as described by user.
    
    total_rewards = values + token_level_rewards
    
    # 2. Compute Returns (Short-term + Long-term)
    # G_t = r_t + gamma * G_{t+1}
    
    returns = torch.zeros_like(total_rewards)
    gen_len = total_rewards.shape[-1]
    last_return = 0.0
    
    # Calculate returns backward
    for t in reversed(range(gen_len)):
        # We need to handle masking. If masked, reward is 0 and next return is 0 (or continued if we want across-mask, but usually not).
        # Assuming values are already masked or irrelevant outside mask.
        # But better be safe with eos_mask.
        
        is_valid = eos_mask[:, t] # (bs,)
        
        current_reward = total_rewards[:, t] * is_valid
        next_return = last_return # (bs,)
        
        # G_t = r_t + gamma * G_{t+1}
        # If current step is invalid (padded), G_t should probably be 0.
        
        current_val = current_reward + gamma * next_return
        returns[:, t] = current_val * is_valid
        
        last_return = current_val
        
    # 3. Group Normalization
    # We normalize the computed Returns at each step across the group.
    # Unlike standard GRPO which normalizes the scalar outcome, here we have dense returns.
    # We normalize per token position? Or per group statistics?
    # Standard GRPO normalizes the *outcome*.
    # Here we have a return G_t for each token.
    # We should probably normalize G_t using the group statistics of G_t.
    
    # However, tokens are aligned? Not necessarily.
    # But for GRPO, we usually compare completions for the same prompt.
    # If we do per-token normalization, we need to handle length differences.
    # But simplified approach: Compute mean/std of G_t for the group *at this step*?
    # No, usually prompts have different lengths, so steps are not aligned.
    
    # Alternative interpretation:
    # Calculate the Total Score for the whole sequence? No, user said "Short term + Long term" which implies dense.
    # Let's normalize G_t using the group's mean/std of G_t *at the same relative position*?
    # Or just normalize using the group's *average return over the sequence*? No.
    
    # Let's try to normalize per token position t.
    # For a group of K responses. At step t.
    # Some responses might have ended.
    # We only normalize among active responses?
    
    # To keep it simple and robust:
    # We can compute the advantage as G_t (unnormalized) or normalized by the whole batch?
    # GRPO specifically means Group Relative.
    # So we must normalize by group.
    
    # Implementation: Iterate over groups, and for each group, normalize returns.
    
    advantages = torch.zeros_like(returns)
    
    id2indices = defaultdict(list)
    bsz = index.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)
        
    for idx, indices in id2indices.items():
        # indices is a list of batch indices belonging to the same group
        group_returns = returns[indices] # (K, T)
        group_mask = eos_mask[indices]   # (K, T)
        
        # We want to normalize G_{k,t} using mean(G_{.,t}) and std(G_{.,t})
        # But only for valid t.
        
        # Mean/Std per time step t
        # (K, T)
        
        # Handle variable lengths:
        # If a sequence has ended, it shouldn't contribute to mean/std of future steps?
        # Or should we treat it as 0? 0 is bad because Return shouldn't be 0 arbitrarily.
        # But here we are just normalizing what we have.
        
        mean = group_returns.sum(dim=0) / (group_mask.sum(dim=0) + epsilon) # (T,)
        # Note: if group_mask.sum is 0, mean is 0.
        
        # Variance
        diff = (group_returns - mean.unsqueeze(0)) * group_mask
        var = (diff ** 2).sum(dim=0) / (group_mask.sum(dim=0) + epsilon)
        std = torch.sqrt(var)
        
        # Normalize
        adv = (group_returns - mean.unsqueeze(0)) / (std.unsqueeze(0) + epsilon)
        
        # Apply mask
        advantages[indices] = adv * group_mask
        
    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
