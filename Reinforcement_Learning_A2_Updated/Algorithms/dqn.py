import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from Models.networks import DQN

class DQNagent:
    def __init__(self, env, learning_rate=0.0005, gamma=0.99, epsilon=1.0, epsilom_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilom_min  # Fixed typo: epsilom_min -> epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return torch.argmax(self.policy_net(state)).item()
        
    def train(self, max_steps=200000):
        reward_records = []  # Store total reward per episode
        step_rewards = []  # Store (step, avg_reward) pairs
        total_steps = 0
        episode = 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if total_steps >= max_steps:
                    break
                
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Compute current Q-value for the action taken
                q_value = self.policy_net(state_tensor)[0, action]
                
                # Compute target Q-values using the same network (naive approach)
                with torch.no_grad():
                    next_q_value = torch.max(self.policy_net(next_state_tensor))
                expected_q = reward + self.gamma * next_q_value * (1 - float(done))
                
                # Calculate loss and update network
                loss = nn.MSELoss()(q_value, expected_q.clone().detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # Record average reward every 1,000 steps
                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0
                    step_rewards.append((total_steps, avg_reward))
            
            if total_steps >= max_steps:
                break

            reward_records.append(episode_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode += 1
            
            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-10:])
                print(f"Steps: {total_steps}, Episode: {episode}, Reward: {episode_reward:.1f}, Avg Reward: {avg_reward:.1f}, Epsilon: {self.epsilon:.3f}")

        return step_rewards  # Return list of (step, avg_reward) pairs
