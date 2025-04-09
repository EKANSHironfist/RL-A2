import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from Models.networks import DQN

class DQNagent:
    def __init__(self, env, learning_rate = 0.0005 , gamma = 0.99, epsilon = 1.0, epsilom_min = 0.01, epsilon_decay = 0.995):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilom_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return torch.argmax(self.policy_net(state)).item()
        
    def train(self, max_steps = 200000):
        all_episodes_rewards = []
        total_steps = 0 
        
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
                
                #Compute current Q-value for the action taken
                q_value = self.policy_net(state_tensor)[0, action]
                
                #Compute target Q-values using the same network (naive approach)
                with torch.no_grad():
                    next_q_value = torch.max(self.policy_net(next_state_tensor))
                expected_q = reward + self.gamma * next_q_value * (1 - float(done))
                
                #Calculate loss and update network
                loss = nn.MSELoss()(q_value, expected_q.clone().detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
            if total_steps > max_steps:
                break 
                
            all_episodes_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if len(all_episodes_rewards) % 10 == 0:
                avg_reward = np.mean(all_episodes_rewards[-10:])
                print(f"Steps: {total_steps}, Episoode: {len(all_episodes_rewards)}, Reward: {episode_reward:.1f}, Avg Reward: {avg_reward:.1f}, Epsilon: {self.epsilon:.3f}")
                
        return all_episodes_rewards