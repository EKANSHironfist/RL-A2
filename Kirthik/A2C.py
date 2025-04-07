import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy network (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Value network (critic)
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, env, gamma=0.99, tau=0.95, entropy_coef=0.005, batch_size=128, update_iters=1):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.entropy_decay = 0.995
        self.min_entropy_coef = 0.001
        self.batch_size = batch_size
        self.update_iters = update_iters

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.value_net = ValueNetwork(self.obs_dim).to(device)

        self.policy_opt = optim.AdamW(self.policy_net.parameters(), lr=1e-4)
        self.value_opt = optim.AdamW(self.value_net.parameters(), lr=5e-4)

        self.reset_buffers()

    def reset_buffers(self):
        self.states, self.actions = [], []
        self.rewards, self.dones = [], []
        self.values, self.next_states = [], []

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_net(state_tensor).squeeze()
        return action.item(), log_prob, entropy.item(), value.item()

    def compute_gae(self, rewards, values, next_values, dones):
        gae = 0
        advs = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
            advs.insert(0, gae)
        return torch.tensor(advs, dtype=torch.float32, device=device)

    def update(self):
        if len(self.states) < self.batch_size:
            return

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)

        next_states_tensor = torch.tensor(np.array(self.next_states), dtype=torch.float32, device=device)
        next_values = self.value_net(next_states_tensor).squeeze().detach()

        advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_iters):
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            policy_loss = -(new_log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
            value_preds = self.value_net(states).squeeze()
            value_loss = F.mse_loss(value_preds, returns.detach())

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

        self.entropy_coef = max(self.entropy_coef * self.entropy_decay, self.min_entropy_coef)
        self.reset_buffers()

    def train(self, total_steps=10**6):
        state, _ = self.env.reset()
        reward_records = []
        avg_rewards = []
        total_reward = 0
        steps = 0

        while steps < total_steps:
            action, log_prob, entropy, value = self.select_action(state)
            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.values.append(value)
            self.next_states.append(next_state)

            total_reward += reward
            steps += 1
            state = next_state if not done else self.env.reset()[0]

            if done:
                reward_records.append(total_reward)
                total_reward = 0

            if len(self.states) >= self.batch_size:
                self.update()

            if steps % 10000 == 0:
                avg = np.mean(reward_records[-50:]) if reward_records else 0
                print(f"Step {steps:7} | Avg Reward (last 50 episodes): {avg:.2f}")
                avg_rewards.append(avg)

        return avg_rewards

# Train the agent
env = gym.make("CartPole-v1")
agent = ActorCriticAgent(env)
rewards = agent.train()

 #Plotting
plt.figure(figsize=(10, 5))
rewards = np.array(rewards)
window_size = 10
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
