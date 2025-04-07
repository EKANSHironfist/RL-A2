import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ActorCriticAgent:
    def __init__(self, env, gamma=0.99, entropy_coef=0.099, batch_size=128):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.entropy_decay = 0.995
        self.min_entropy_coef = 0.01  # Increased minimum to maintain exploration
        self.batch_size = batch_size

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.value_net = ValueNetwork(self.obs_dim).to(device)

        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=7e-4)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=1e-4)  # Matched with policy

        self.reset_buffers()

    def reset_buffers(self):
        self.states, self.actions = [], []
        self.rewards, self.dones = [], []
        self.log_probs, self.values = [], []
        self.entropies = []
        self.episode_starts = [0]  # Track start indices of episodes

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        state_value = self.value_net(state_tensor)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return action.item(), log_prob, state_value, entropy

    def compute_monte_carlo_returns(self, rewards, dones):
        returns = []
        episode_returns = []
        R = 0

        # Compute returns for each episode in the batch
        for i in range(len(rewards)):
            r = rewards[i]
            done = dones[i]
            R = r + self.gamma * R
            episode_returns.append(R)

            if done or i == len(rewards) - 1:  # End of episode or batch
                # Reverse and compute returns for this episode
                episode_returns = episode_returns[::-1]
                returns.extend(episode_returns)
                episode_returns = []
                R = 0

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return returns

    def update(self):
        if len(self.states) < self.batch_size:
            return

        # Convert buffers to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        entropies = torch.stack(self.entropies)
        rewards = self.rewards
        dones = self.dones

        # Compute Monte Carlo returns per episode
        returns = self.compute_monte_carlo_returns(rewards, dones)

        # Calculate advantages
        advantages = returns - values.squeeze(-1).detach()

        # Update policy (actor)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        actor_loss = policy_loss + entropy_loss

        self.policy_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)  # Gradient clipping
        self.policy_opt.step()

        # Update value function (critic)
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)  # Gradient clipping
        self.value_opt.step()

        # Decay entropy coefficient
        self.entropy_coef = max(self.entropy_coef * self.entropy_decay, self.min_entropy_coef)

        # Reset buffers
        self.reset_buffers()

    def train(self, total_steps=10**6):
        state, _ = self.env.reset()
        reward_records = []
        avg_rewards = []
        total_reward = 0
        steps = 0

        while steps < total_steps:
            action, log_prob, state_value, entropy = self.select_action(state)
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            # Store data in buffers
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.log_probs.append(log_prob)
            self.values.append(state_value)
            self.entropies.append(entropy)

            if done:
                self.episode_starts.append(len(self.states))  # Mark episode boundary

            total_reward += reward
            steps += 1
            state = next_state if not done else self.env.reset()[0]

            if done:
                reward_records.append(total_reward)
                total_reward = 0

            # Update after collecting a batch
            if len(self.states) >= self.batch_size:
                self.update()

            if steps <= 50000:  # For the first 50,000 steps, record every 1,000 steps
                if steps % 1000 == 0:
                    avg = np.mean(reward_records[-50:]) if reward_records else 0
                    print(f"Step {steps:7} | Avg Reward (last 50 episodes): {avg:.2f}")
                    avg_rewards.append(avg)
            else:  # After 50,000 steps, record every 10,000 steps
                if steps % 10000 == 0:
                    avg = np.mean(reward_records[-50:]) if reward_records else 0
                    print(f"Step {steps:7} | Avg Reward (last 50 episodes): {avg:.2f}")
                    avg_rewards.append(avg)

        return avg_rewards

# Train the agent
env = gym.make("CartPole-v1")
agent = ActorCriticAgent(env)
rewards = agent.train()

# Plotting
plt.figure(figsize=(10, 5))
rewards = np.array(rewards)
window_size = 20
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
x = np.linspace(0, 1_000_000, len(smoothed_rewards)) / 100_000
rolling_std = np.zeros(len(smoothed_rewards))
for i in range(len(smoothed_rewards)):
    start = max(0, i - window_size // 2)
    end = min(len(rewards), i + window_size // 2 + 1)
    rolling_std[i] = np.std(rewards[start:end])

plt.plot(x, smoothed_rewards, color='#4682B4', linewidth=2.5, label='Smoothed Avg Reward')
plt.fill_between(x, smoothed_rewards - rolling_std, smoothed_rewards + rolling_std,
                 color='#4682B4', alpha=0.2, label='±1 Std Dev')
plt.title("A2C on CartPole-v1", fontsize=14, pad=15)
plt.xlabel("Training Steps (x100k steps)", fontsize=12)
plt.ylabel("Average Reward (last 50 episodes)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 550)
plt.grid(True, linestyle='--', alpha=0.4, color='gray')
plt.gca().set_facecolor('#F5F5F5')
plt.gcf().set_facecolor('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_color('gray')
plt.legend(loc='best', fontsize=10, framealpha=0.8, facecolor='white', edgecolor='gray')
plt.show()