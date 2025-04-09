# Utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(all_results, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    def smooth_curve(points, factor=0.9):
        smoothed = []
        last = points[0]
        for point in points:
            last = last * factor + (1 - factor) * point
            smoothed.append(last)
        return smoothed

    plt.figure(figsize=(12, 6))

    for algo_name, rewards in all_results.items():
        # Make all reward lists the same length
        min_len = min(len(run) for run in rewards)
        for i in range(len(rewards)):
            rewards[i] = rewards[i][:min_len]
        
        rewards = np.array(rewards)  # shape: [runs, episodes]
        
        if len(rewards) < 2:
            print(f"Skipping {algo_name}: Not enough data to plot.")
            continue

        # Fix here: average across runs
        mean_rewards = np.mean(rewards, axis=0)
        smoothed_mean = smooth_curve(mean_rewards)

        x = np.arange(len(smoothed_mean))
        plt.plot(x, smoothed_mean, label=algo_name)


    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
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
    plt.ylabel('Average Reward')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    
