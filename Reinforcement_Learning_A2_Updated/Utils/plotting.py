# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(all_results, filename):
    plt.figure(figsize=(12, 6))

    for algo_name, step_rewards_list in all_results.items():
        # step_rewards_list is a list of runs, where each run is a list of (step, avg_reward) pairs
        # Interpolate to a common set of steps for averaging across runs
        max_steps = max(step for run in step_rewards_list for step, _ in run)
        common_steps = np.arange(0, max_steps + 1000, 1000)  # Steps at intervals of 1,000
        
        # Interpolate rewards for each run at common steps
        interpolated_rewards = []
        for run in step_rewards_list:
            steps, rewards = zip(*run)  # Unpack (step, avg_reward) pairs
            steps = np.array(steps)
            rewards = np.array(rewards)
            # Interpolate rewards at common_steps
            interpolated = np.interp(common_steps, steps, rewards)
            interpolated_rewards.append(interpolated)
        
        interpolated_rewards = np.array(interpolated_rewards)  # Shape: [runs, num_steps]
        
        if len(interpolated_rewards) < 2:
            print(f"Skipping {algo_name}: Not enough data to plot.")
            continue

        # Average across runs
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        
        # Smooth the mean rewards
        def smooth_curve(points, factor=0.9):
            smoothed = []
            last = points[0]
            for point in points:
                last = last * factor + (1 - factor) * point
                smoothed.append(last)
            return smoothed
        
        smoothed_mean = smooth_curve(mean_rewards)
        
        # Plot: x-axis in units of 100k steps
        x = common_steps / 100000  # Convert steps to 100k steps
        plt.plot(x, smoothed_mean, label=algo_name)
        plt.fill_between(x, smoothed_mean - std_rewards, smoothed_mean + std_rewards, alpha=0.2)

    plt.xlabel("Training Steps (x100K steps)")
    plt.ylabel("Average Reward (last 50 episodes)")
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
    
    for algo_name, step_rewards_list in results_dict.items():
        # Take the last reward value from each run
        final_performances = [run[-1][1] for run in step_rewards_list]  # run[-1][1] is the last avg_reward
        data.append(final_performances)
        labels.append(algo_name)
    
    plt.boxplot(data, labels=labels)
    plt.ylabel('Average Reward')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
   
