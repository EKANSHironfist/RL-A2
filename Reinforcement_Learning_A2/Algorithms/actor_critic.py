import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Models.networks import PolicyNetwork, ValueNetwork

class ActorCritic:
    def __init__(self, env, learning_rate = 0.001, gamma = 0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Actor (policy) Network
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)
        
        #Critic (value) Network
        self.value_net = ValueNetwork(self.state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr = learning_rate)
        
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        state_value = self.value_net(state)
        return action.item(), log_prob, state_value
    
    def train(self, num_episodes = 500):
        all_episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            episode_reward = 0
            done = False
            
            #Collect one step trajectory and update
            while not done:
                action, log_prob, state_value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                values.append(state_value)
                rewards.append(reward)
                episode_reward += reward
                
                #Get next state value for TD target
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_state_value  = self.value_net(next_state_tensor) if not done else torch.tensor([0.0]).to(self.device)
                
                #TD target
                td_target = reward + self.gamma * next_state_value.detach()
                
                #TD error (used as advantage)
                td_error = td_target - state_value
                
                #Update value networks
                value_loss = F.mse_loss(state_value.view(-1), td_target.detach().view(-1))
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                #Update policy network
                policy_loss = -log_prob * td_error.detach()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                state = next_state
                
            all_episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.1f}")
                
        return all_episode_rewards
                