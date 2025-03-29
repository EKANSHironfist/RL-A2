# Utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(results_dict, filename, window_size=20):
    plt.figure(figsize=(12, 6))
    
    # Smoothing function
    def smooth(y, window_size):
        box = np.ones(window_size) / window_size
        return np.convolve(y, box, mode='same')
    
    # Plot each algorithm's results
    for algo_name, rewards_list in results_dict.items():
        # If multiple runs, calculate mean and std dev
        if isinstance(rewards_list[0], list):
            # Pad shorter runs
            max_len = max(len(rewards) for rewards in rewards_list)
            padded_rewards = [np.pad(rewards, (0, max_len - len(rewards)), mode='constant', constant_values=np.nan) for rewards in rewards_list]
            
            mean_rewards = np.nanmean(padded_rewards, axis=0)
            std_rewards = np.nanstd(padded_rewards, axis=0)
            
            x = np.arange(len(mean_rewards))
            smoothed_mean = smooth(mean_rewards, window_size)
            
            plt.plot(x, smoothed_mean, label=algo_name)
            plt.fill_between(x, smoothed_mean - std_rewards, smoothed_mean + std_rewards, alpha=0.2)
        else:
            # Single run
            smoothed = smooth(rewards_list, window_size)
            plt.plot(smoothed, label=algo_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    plt.show()

def plot_comparison_boxplot(results_dict, filename):  
    plt.figure(figsize=(10, 6))
    
    # Extract final performance data
    data = []
    labels = []
    
    for algo_name, rewards_list in results_dict.items():
        if isinstance(rewards_list[0], list):
            final_performances = [np.mean(run[-len(run)//10:]) for run in rewards_list]
        else:
            final_performances = [np.mean(rewards_list[-len(rewards_list)//10:])]
        
        data.append(final_performances)
        labels.append(algo_name)
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Average Reward (Final 10% of Episodes)')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    plt.show()