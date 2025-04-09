# reinforce.py
import torch
import torch.optim as optim
import numpy as np
from Models.networks import PolicyNetwork

class REINFORCE:
    def __init__(self, env, learning_rate=0.0005, gamma=0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy Network
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Discount Factor
        self.gamma = gamma
        
    def select_action(self, state):
        # State to tensor and action probs
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        # Action Sample
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    # Monte Carlo estimation of Q-Values
    def calculate_returns(self, rewards):
        returns = []
        R = 0
        
        # Calculate returns from the end of episode to the beginning
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
    
    def train(self, max_steps=200000):
        reward_records = []  # Store total reward per episode
        step_rewards = []  # Store (step, avg_reward) pairs
        total_steps = 0
        episode = 0
        
        while total_steps < max_steps:
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            episode_reward, done = 0, False
            
            while not done:
                if total_steps >= max_steps:
                    break
                
                # Collect trajectory (Monte Carlo Method)
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward
                state = next_state
                total_steps += 1
                
                # Record average reward every 1,000 steps
                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0
                    step_rewards.append((total_steps, avg_reward))
                
            if total_steps >= max_steps:
                break    
                
            # Calculate return
            returns = self.calculate_returns(rewards)
            
            # Calculate loss and update policy
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)  # Negative for gradient ascent
                
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
            
            reward_records.append(episode_reward)
            episode += 1
            
            # Log Progress
            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-10:])
                print(f"Steps: {total_steps}, Episode: {episode}, Avg Reward: {avg_reward:.1f}")
        
        return step_rewards  # Return list of (step, avg_reward) pairs
