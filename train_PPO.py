#!/usr/bin/env python3
import argparse
import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2

# ---------------- Agent Definition ----------------
class PPOAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()
        # Actor network
        actor_hid1_size = num_inputs * 20
        actor_hid3_size = num_actions * 10
        actor_hid2_size = int(np.sqrt(actor_hid1_size * actor_hid3_size))
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, actor_hid1_size), nn.Tanh(),
            nn.Linear(actor_hid1_size, actor_hid2_size), nn.Tanh(),
            nn.Linear(actor_hid2_size, actor_hid3_size), nn.Tanh(),
            nn.Linear(actor_hid3_size, num_actions), nn.Tanh()  # outputs in [-1,1]
        )
        # Log std parameter
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)
        # Critic network
        critic_hid1_size = num_inputs * 20
        critic_hid3_size = 10
        critic_hid2_size = int(np.sqrt(critic_hid1_size * critic_hid3_size))
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, critic_hid1_size), nn.Tanh(),
            nn.Linear(critic_hid1_size, critic_hid2_size), nn.Tanh(),
            nn.Linear(critic_hid2_size, critic_hid3_size), nn.Tanh(),
            nn.Linear(critic_hid3_size, 1)
        )

    def forward(self, x: torch.Tensor):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_value(self, x: torch.Tensor):
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(x).squeeze(-1)
        return action, log_prob, entropy, value

# ---------------- Buffer Definition ----------------
class PPOBuffer:
    def __init__(
        self, obs_dim, act_dim, size: int, num_envs: int, device,
        gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        self.capacity = size
        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0

    def store(self, obs, act, rew, val, term, trunc, logprob):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob
        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        assert self.ptr == self.capacity, "Buffer not full"
        adv_buf = torch.zeros_like(self.rew_buf)
        last_gae = 0.0
        for t in reversed(range(self.capacity)):
            next_vals = last_vals if t == self.capacity - 1 else self.val_buf[t + 1]
            term_mask = (1.0 - last_terminateds) if t == self.capacity - 1 else (1.0 - self.term_buf[t + 1])
            trunc_mask = (1.0 - last_truncateds) if t == self.capacity - 1 else (1.0 - self.trunc_buf[t + 1])
            delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
            last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + self.val_buf
        return adv_buf, ret_buf

    def get(self):
        assert self.ptr == self.capacity, "Buffer not full"
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.logprob_buf

# ---------------- Utilities ----------------
def parse_args_ppo() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=(torch.cuda.is_available()), help="Use CUDA if available")
    parser.add_argument("--env", default="HalfCheetah-v5", help="Gym environment ID")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--n-epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per epoch per env")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size for PPO update")
    parser.add_argument("--train-iters", type=int, default=20, help="PPO update iterations per epoch")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.1, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=1e-4, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--reward-scale", type=float, default=0.005, help="Reward scaling factor")
    parser.add_argument("--render-epoch", type=int, default=25, help="Render every n epochs")
    return parser.parse_args()


def make_env(env_id, reward_scaling=0.01, render=False, fps=30):
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
    else:
        env = gym.make(env_id)
    return gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)


def log_video(env, agent, device, video_path, fps=30):
    agent.eval()
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            tensor_obs = torch.tensor(np.array([obs], dtype=np.float32), device=device)
            action, _, _, _ = agent.get_action_and_value(tensor_obs)
        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
        done = terminated or truncated
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# ---------------- PPO Update ----------------
def ppo_update(
    agent, optimizer, scaler,
    batch_obs, batch_actions, batch_returns,
    batch_old_log_probs, batch_adv,
    clip_epsilon, vf_coef, ent_coef
):
    agent.train()
    optimizer.zero_grad()
    _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs, batch_actions)
    ratio = torch.exp(new_log_probs - batch_old_log_probs)
    batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
    surr1 = ratio * batch_adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.MSELoss()(new_values.squeeze(-1), batch_returns)
    entropy = entropies.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

# ---------------- Main Training Loop ----------------
def main():
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Logging setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    videos_dir = os.path.join(base_dir, "videos", ts)
    ckpt_dir = os.path.join(base_dir, "checkpoints", ts)
    log_dir = os.path.join(base_dir, "logs", ts)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    writer.add_text("hyperparams", "\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    # Environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)]
    )
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    # Agent & optimizer
    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: (e / 10) if e < 10 else 0.999 ** (e - 10)
    )
    scaler = torch.cuda.amp.GradScaler()

    # Buffer
    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    # Training
    next_obs = torch.tensor(envs.reset()[0], dtype=torch.float32, device=device)
    next_terminateds = torch.zeros(args.n_envs, device=device)
    next_truncateds = torch.zeros(args.n_envs, device=device)
    reward_history = []

    best_mean_reward = -np.inf
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.n_epochs + 1):
        # Rollout
        for _ in tqdm(range(args.n_steps), desc=f"Epoch {epoch} rollout"):
            global_step += args.n_envs
            obs = next_obs
            term = next_terminateds
            trunc = next_truncateds
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs)
            actions_np = actions.cpu().numpy()
            next_obs_np, rewards_np, next_term_np, next_trunc_np, _ = envs.step(actions_np)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
            next_terminateds = torch.tensor(next_term_np, dtype=torch.float32, device=device)
            next_truncateds = torch.tensor(next_trunc_np, dtype=torch.float32, device=device)
            reward_history.extend(rewards_np.tolist())
            buffer.store(obs, actions, rewards, values, term, trunc, logprobs)

        # Compute advantages and returns
        with torch.no_grad():
            # next_obs: (n_envs, obs_dim)
            last_vals = agent.get_value(next_obs)       # -> (n_envs, 1)
            last_vals = last_vals.squeeze(-1)           # -> (n_envs,)
        adv, ret = buffer.calculate_advantages(last_vals, next_terminateds, next_truncateds)
        obs_buf, act_buf, logp_buf = buffer.get()
        adv = adv.view(-1)
        ret = ret.view(-1)
        obs_flat = obs_buf.view(-1, obs_dim[0])
        act_flat = act_buf.view(-1, act_dim[0])
        logp_flat = logp_buf.view(-1)

        # PPO updates
        for _ in range(args.train_iters):
            idxs = np.random.permutation(len(obs_flat))
            for start in range(0, len(obs_flat), args.batch_size):
                mb_idx = idxs[start:start+args.batch_size]
                loss, pl, vl, ent = ppo_update(
                    agent, optimizer, scaler,
                    obs_flat[mb_idx].to(device), act_flat[mb_idx].to(device),
                    ret[mb_idx].to(device), logp_flat[mb_idx].to(device), adv[mb_idx].to(device),
                    args.clip_ratio, args.vf_coef, args.ent_coef
                )
        scheduler.step()

        # Logging
        mean_reward = float(np.mean(reward_history) / args.reward_scale)
        writer.add_scalar("reward/mean", mean_reward, epoch)
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
        reward_history.clear()
        print(f"Epoch {epoch} | Mean Reward: {mean_reward:.2f} | Time: {time.time()-start_time:.2f}s")
        start_time = time.time()

        # Save best and last models
        model_path = ckpt_dir if (mean_reward <= best_mean_reward) else os.path.join(ckpt_dir, "best.pt")
        if mean_reward > best_mean_reward:
            torch.save(agent.state_dict(), model_path)
            best_mean_reward = mean_reward

        # Video logging
        if epoch % args.render_epoch == 0:
            log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

    envs.close()
    test_env.close()
    writer.close()

if __name__ == "__main__":
    main()
