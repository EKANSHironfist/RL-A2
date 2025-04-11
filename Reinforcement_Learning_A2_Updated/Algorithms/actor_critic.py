import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Models.networks import PolicyNetwork, ValueNetwork

class ActorCritic:
    def __init__(self, env, learning_rate=0.002, gamma=0.99):  
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        
        self.value_net = ValueNetwork(self.state_dim)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        state_value = self.value_net(state)  # Outputs V(s)
        return action.item(), log_prob, state_value
    
    def train(self, max_steps=200000):
        reward_records = []  
        step_rewards = [] 
        total_steps = 0
        episode = 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            log_probs, values, rewards = [], [], []
            episode_reward, done = 0, False

            while not done: 
                if total_steps >= max_steps:
                    break
                
                action, log_prob, value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                total_steps += 1
                episode_reward += reward
                state = next_state
                
                
                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0
                    step_rewards.append((total_steps, avg_reward))
            
            if total_steps >= max_steps:
                break

            # Compute Monte-Carlo returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            # Normalize returns
            returns = torch.tensor(returns).float()
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  

            log_probs = torch.cat(log_probs)
            values = torch.cat(values)  # Shape: [N, 1]

            
            values = values.squeeze(-1)  
            policy_loss = -(log_probs * returns.detach()).mean()  # Use normalized returns
            value_loss = F.mse_loss(values, returns)

            # Debugging prints
            if episode % 10 == 0:
                print(f"Episode: {episode}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Avg Return: {returns.mean().item():.1f}")

            # Backprop (no gradient clipping)
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

            reward_records.append(episode_reward)
            episode += 1

            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-50:])  
                print(f"Steps: {total_steps}, Episode: {episode}, Avg Reward: {avg_reward:.1f}")

        return step_rewards  
