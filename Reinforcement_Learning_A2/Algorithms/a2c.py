import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Models.networks import PolicyNetwork, ValueNetwork

class A2C:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor (policy) Network
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Critic (value) Network
        self.value_net = ValueNetwork(self.state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        state_value = self.value_net(state)
        return action.item(), log_prob, state_value
    
    def train(self, num_episodes=500):
        all_episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            entropies = []
            episode_reward = 0
            done = False
            
            # Collect trajectory for one episode
            while not done:
                action, log_prob, state_value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                values.append(state_value)
                rewards.append(reward)
                episode_reward += reward
                
                # Calculate entropy for exploration bonus
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                probs = self.policy_net(state_tensor)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropies.append(entropy)
                
                state = next_state
                
            # Process episode data
            returns = []
            R = 0 
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns).float().to(self.device)
            
            # Normalize return
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            entropies = torch.stack(entropies)
            
            # Calculate advantages
            # Solution 1: Squeeze values to match returns shape
            advantages = returns - values.squeeze(-1).detach()
            
            # Update Policy (actor)
            policy_loss = -(log_probs * advantages.detach()).mean()
            entropy_loss = -0.01 * entropies.mean()  # Entropy bonus for exploration
            actor_loss = policy_loss + entropy_loss
            
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()
            
            # Update value function (critic)
            # Solution 1: Squeeze values to match returns shape
            value_loss = F.mse_loss(values.squeeze(-1), returns)
            
            # Alternative Solution 2: Unsqueeze returns to match values shape
            # value_loss = F.mse_loss(values, returns.unsqueeze(-1))
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            all_episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Episode {episode}, Avg Rewards (last 10): {avg_reward:.1f}")
                
        return all_episode_rewards