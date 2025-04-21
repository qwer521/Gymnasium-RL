#!/usr/bin/env python3
import os
import random
import numpy as np
from collections import deque
from datetime import datetime
from PIL import Image
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import ale_py

# --------------------- Hyperparameters ---------------------
ENV_ID             = "ALE/BeamRider-v5"
SEED               = 42
NUM_EPISODES       = 1000000
BATCH_SIZE         = 32
GAMMA              = 0.99
LR                 = 1e-4
BUFFER_SIZE        = 100_000
TARGET_UPDATE      = 1_000        # steps between target net updates
FRAME_STACK_K      = 4
SAVE_INTERVAL      = 25          # episodes between model checkpoints and video captures
MOVING_AVG_W       = 10           # window for moving average plot

# ε‑greedy schedule
eps_start_init     = 1.0
eps_restart_init   = eps_start_init
eps_end            = 0.1
eps_decay          = 100_000
restart_decay      = 0.9          # factor to decay eps_restart
patience_episodes  = 50           # episodes without improvement

# --------------------- Device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Timestamped Directories ---------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR = os.path.join("checkpoints", timestamp)
VIDEO_DIR      = os.path.join("videos",    timestamp)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR,      exist_ok=True)

# --------------------- Utilities ---------------------
def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        proc = self._preprocess(obs)
        for _ in range(self.k): self.frames.append(proc)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        proc = self._preprocess(obs)
        self.frames.append(proc)
        return np.stack(self.frames, axis=0)

    @staticmethod
    def _preprocess(obs):
        img = Image.fromarray(obs).convert("L").resize((81, 81))
        return np.array(img, dtype=np.uint8)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        if len(self.buffer) < self.capacity: self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        ss, aa, rr, ss2, dd = zip(*batch)
        states      = torch.from_numpy(np.stack(ss,  axis=0)).float().to(device)      / 255.0
        next_states = torch.from_numpy(np.stack(ss2, axis=0)).float().to(device) / 255.0
        actions     = torch.tensor(aa, dtype=torch.int64, device=device)
        rewards     = torch.tensor(rr, dtype=torch.float32, device=device)
        dones       = torch.tensor(dd, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, in_ch, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),    nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),    nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 81, 81)
            conv_out = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

# --------------------- Plot Utility ---------------------
def save_plot(rewards, window, episode):
    mov_avg = [np.mean(rewards[max(0,i-window):i]) for i in range(1,len(rewards)+1)]
    plt.figure(figsize=(6,4))
    plt.plot(rewards, label="Reward")
    plt.plot(mov_avg,   label=f"{window}-Ep MA")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.title(f"Up to Ep {episode}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "figure.png")); plt.close()

# --------------------- Main Training ---------------------
def main():
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join("logs", timestamp))

    # Environment with video recording
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        base_env,
        video_folder=VIDEO_DIR,
        name_prefix="beamrider",
        episode_trigger=lambda ep: ep % SAVE_INTERVAL == 0
    )
    set_seed(env, SEED)

    fs = FrameStack(FRAME_STACK_K)
    policy_net = DQN(FRAME_STACK_K, env.action_space.n).to(device)
    target_net = DQN(FRAME_STACK_K, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict()); target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = ReplayBuffer(BUFFER_SIZE)

    steps_done        = 0
    episode_rewards   = []
    eps_start         = eps_start_init
    eps_restart       = eps_restart_init
    last_reset_step   = 0
    epsilon_min_flag  = False
    best_reward_min   = -np.inf
    no_improve_count  = 0

    for ep in range(1, NUM_EPISODES+1):
        # decay restart ε periodically
        if ep % SAVE_INTERVAL == 0:
            eps_restart *= restart_decay
            print(f"[Auto-decay] Ep {ep}: eps_restart={eps_restart:.3f}")

        obs, _ = env.reset()
        state  = fs.reset(obs); total_reward = 0; done=False

        while not done:
            eps = eps_end + (eps_start-eps_end)*np.exp(-1.*(steps_done-last_reset_step)/eps_decay)
            if not epsilon_min_flag and eps <= eps_end+1e-6:
                epsilon_min_flag=True; best_reward_min=-np.inf; no_improve_count=0
                print(f"[Min ε reached] Ep {ep}")

            steps_done += 1
            # select action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                s_t = torch.from_numpy(state[None]).float().to(device)/255.0
                with torch.no_grad(): action = policy_net(s_t).argmax(1).item()

            nxt_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_state = fs.step(nxt_obs)

            memory.push(state, action, reward, next_state, done)
            state = next_state; total_reward += reward

            # optimize
            if len(memory) >= BATCH_SIZE:
                s,a,r,s2,d = memory.sample(BATCH_SIZE)
                q_pred = policy_net(s).gather(1,a.unsqueeze(1)).squeeze(1)
                next_a = policy_net(s2).argmax(1,keepdim=True)
                q_next = target_net(s2).gather(1,next_a).squeeze(1)
                q_target = r + GAMMA*q_next*(1-d)
                loss = F.mse_loss(q_pred, q_target.detach())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                writer.add_scalar("Loss/TD", loss.item(), steps_done)

            # update target
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        print(f"Ep {ep:4d} | Reward: {total_reward:6.1f} | ε: {eps:.3f}")

        # log to TensorBoard
        writer.add_scalar("Reward/Episode", total_reward, ep)
        writer.add_scalar("Epsilon/Value", eps, ep)
        mov_avg = np.mean(episode_rewards[-MOVING_AVG_W:])
        writer.add_scalar(f"Reward/Ep_MA", mov_avg, ep)

        # patience-based reset
        if epsilon_min_flag:
            if total_reward > best_reward_min:
                best_reward_min = total_reward; no_improve_count=0
            else:
                no_improve_count += 1
            if no_improve_count >= patience_episodes:
                eps_start = eps_restart; last_reset_step = steps_done; epsilon_min_flag=False
                print(f"[Patience reset] Ep {ep}: ε reset to {eps_start:.3f}")

        # plot & checkpoint
        save_plot(episode_rewards, MOVING_AVG_W, ep)
        torch.save(policy_net.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pth"))

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
