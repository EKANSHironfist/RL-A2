import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym

# Define Policy and Value Networks (unchanged)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic:
    def __init__(self, env, policy_lr=5e-4, value_lr=2e-4, gamma=0.99, entropy_coef=0.05, batch_size=2048):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.value_net = ValueNetwork(self.state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        # Buffers for batch processing
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        state_value = self.value_net(state)
        return action.item(), log_prob, entropy, state_value
    
    def update_networks(self):
        if len(self.states) < self.batch_size:
            return None, None
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        log_probs = torch.cat(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Compute values and advantages
        state_values = self.value_net(states).squeeze(-1)
        next_state_values = self.value_net(next_states).squeeze(-1)
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = (td_targets - state_values).detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update value network
        value_loss = F.mse_loss(state_values, td_targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        # Update policy network
        probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(probs)
        entropy = action_dist.entropy().mean()
        policy_loss = (-log_probs * advantages - self.entropy_coef * entropy).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        return value_loss.item(), policy_loss.item()
    
    def train(self, num_steps=10**6, print_interval=10000):
        state, _ = self.env.reset()
        total_reward = 0
        episode_rewards = []
        rewards_per_step = []
        step_count = 0
        last_value_loss = 0
        last_policy_loss = 0
        
        while step_count < num_steps:
            episode_reward = 0
            done = False
            
            while not done and step_count < num_steps:
                action, log_prob, entropy, state_value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(float(done))
                
                episode_reward += reward
                total_reward += reward
                rewards_per_step.append(total_reward)
                step_count += 1
                
                # Update networks when batch is full
                if len(self.states) >= self.batch_size:
                    value_loss, policy_loss = self.update_networks()
                    if value_loss is not None:
                        last_value_loss = value_loss
                        last_policy_loss = policy_loss
                
                state = next_state if not done else self.env.reset()[0]
            
            episode_rewards.append(episode_reward)
            
            if len(episode_rewards) > 100:
                moving_avg_reward = np.mean(episode_rewards[-100:])
            else:
                moving_avg_reward = np.mean(episode_rewards)
            
            if step_count % print_interval < 1000:  # Print once per interval
                print(f"Step {step_count}, Moving Avg Reward (100 episodes): {moving_avg_reward:.2f}, Value Loss: {last_value_loss}, Policy Loss: {last_policy_loss}")
        
        return episode_rewards

# Plot function (unchanged)
def plot_learning_curve(rewards, filename="learning_curve.png", window_size=100):
    plt.figure(figsize=(12, 6))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards, label="Moving Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Example usage
env = gym.make("CartPole-v1")
ac_agent = ActorCritic(env, policy_lr=5e-4, value_lr=2e-4, gamma=0.99, entropy_coef=0.05, batch_size=2048)
rewards = ac_agent.train()
plot_learning_curve(rewards)