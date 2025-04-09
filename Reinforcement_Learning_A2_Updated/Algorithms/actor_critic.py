import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Models.networks import PolicyNetwork, ValueNetwork

class ActorCritic:
    def __init__(self, env, learning_rate=0.0005, gamma=0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        
        # Initialize networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value_net = ValueNetwork(self.state_dim).to(self.device)  # Fix: Added output_dim=1
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        state_value = self.value_net(state)
        return action.item(), log_prob, state_value
    
    def train(self, max_steps = 200000):
        all_episode_rewards = []
        total_steps = 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            episode_reward, done = 0, False

            # Collect trajectory
            while not done:
                if total_steps >= max_steps:
                    break
                
                # Collect trajectory
                action, log_prob, value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                episode_reward += reward
                state = next_state
                total_steps += 1
                
                if total_steps > max_steps:
                    break

            # Process trajectory
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns).to(self.device)

            # Compute losses
            value_losses = []
            policy_losses = []

            for log_prob, value, R in zip(log_probs, values, returns):
                target = torch.tensor([[R]], device=self.device)  # ensure shape [1,1]
                value_losses.append(F.mse_loss(value, target))

                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)

            value_loss = torch.stack(value_losses).sum()
            policy_loss = torch.stack(policy_losses).sum()
            total_loss = value_loss + policy_loss

            # Backprop all at once
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            all_episode_rewards.append(episode_reward)

            if len(all_episode_rewards) % 10 == 0:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Steps: {total_steps}, Episode: {len(all_episode_rewards)}, Avg Reward: {avg_reward:.1f}")

        return all_episode_rewards   