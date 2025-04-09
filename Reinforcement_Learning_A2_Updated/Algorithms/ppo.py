import torch
import torch.nn.functional as F
import numpy as np
from Models.networks import PolicyNetwork, ValueNetwork
import torch.optim as optim

class PPO:
    def __init__(self, env, lr_policy=0.0005, lr_value=0.0005, gamma=0.99, clip_eps=0.2):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.value_net = ValueNetwork(env.observation_space.shape[0]).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.clip_eps = clip_eps
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def compute_returns(self, rewards, masks):
        returns = []
        R = 0
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def train(self, max_steps=200000, batch_size=5):
        reward_records = []  # Store total reward per episode
        step_rewards = []  # Store (step, avg_reward) pairs
        total_steps = 0
        episode = 0

        while total_steps < max_steps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            states, actions, rewards, masks, log_probs = [], [], [], [], []
            episode_steps = 0

            while not done:
                if total_steps >= max_steps:
                    break
                
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                masks.append(1 - float(done))
                log_probs.append(log_prob)

                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_steps += 1
                
                # Record average reward every 1,000 steps
                if total_steps % 1000 == 0:
                    avg_reward = np.mean(reward_records[-50:]) if reward_records else 0
                    step_rewards.append((total_steps, avg_reward))
            
            if total_steps >= max_steps:
                break

            # Perform PPO update
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)
            returns = torch.FloatTensor(self.compute_returns(rewards, masks)).to(self.device)

            for _ in range(batch_size):
                values = self.value_net(states).squeeze(-1)
                advantages = returns - values.detach()

                action_probs = self.policy_net(states)
                dist = torch.distributions.Categorical(action_probs)
                curr_log_probs = dist.log_prob(actions)

                ratios = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

            reward_records.append(episode_reward)
            episode += 1

            # Log Progress
            if episode % 10 == 0:
                avg_reward = np.mean(reward_records[-10:])
                print(f"Steps: {total_steps}, Episode: {episode}, Avg Reward: {avg_reward:.1f}")

        return step_rewards 
