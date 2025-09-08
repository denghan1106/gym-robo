import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import Actor, Critic

class PPO:
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_eps=0.2, lr=3e-4, K_epochs=10):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_mean, std = self.actor(obs_tensor)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        return action.squeeze().numpy(), dist.log_prob(action).sum().item()

    def store_transition(self, s, a, r, d, logp, next_s):
        self.buffer.append((s, a, r, d, logp, next_s))

    def update(self):
        states, actions, rewards, dones, logps, next_states = zip(*self.buffer)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        old_logps = torch.FloatTensor(logps)

        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        advantages = returns - self.critic(states).squeeze().detach()

        for _ in range(self.K_epochs):
            # Actor update
            action_mean, std = self.actor(states)
            dist = torch.distributions.Normal(action_mean, std)
            new_logps = dist.log_prob(actions).sum(axis=-1)
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            # Critic update (separate from actor to avoid graph conflicts)
            critic_values = self.critic(states).squeeze()
            loss_critic = nn.MSELoss()(critic_values, returns)

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

        self.buffer = []
