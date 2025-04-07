# import gymnasium as gym
# import torch
# import random
# import numpy as np
# import argparse
# import os
# from Algorithms.reinforce import REINFORCE
# from Algorithms.actor_critic import ActorCritic
# from Algorithms.a2c import A2C
# from Algorithms.dqn import DQNagent
# from Utils.plotting import plot_learning_curves, plot_comparison_boxplot

# def set_seeds(seed = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
        
# def run_algorithm(algo_name, num_runs=5 , num_episodes=500, seed=42): # previously it was 5 runs
#     results = []
    
#     # Map input algorithm names to standard formats
#     algo_map = {
#         "reinforce": "REINFORCE",
#         "actor_critic": "ActorCritic",
#         "a2c": "A2C",
#         "dqn": "DQNAgent"
#     }
    
#     # Convert the algorithm name to a standard format
#     standard_name = algo_map.get(algo_name.lower())
#     if standard_name:
#         algo_name = standard_name
    
#     for run in range(num_runs):
#         print(f"\nRunning {algo_name}, Run {run+1}/{num_runs}")
#         set_seeds(seed + run)
#         env = gym.make("CartPole-v1")
        
#         if algo_name == "REINFORCE":
#             agent = REINFORCE(env)#learning_rate=0.001, gamma=0.99)
#         elif algo_name == "ActorCritic":
#             agent = ActorCritic(env, learning_rate=0.001, gamma=0.99)
#         elif algo_name == "A2C":
#             agent = A2C(env, learning_rate=0.001, gamma=0.99)
#         elif algo_name == "DQNAgent":
#             agent = DQNagent(env)
#         else:
#             raise ValueError(f"Unknown Algorithm: {algo_name}")
        
#         rewards = agent.train(num_episodes=num_episodes)
#         results.append(rewards)
#         env.close()
    
#     # Save Results
#     os.makedirs('results', exist_ok=True)
#     np.save(f"results/{algo_name}_rewards.npy", results)
#     return results

# def run_all_algorithms(num_runs = 1, num_episodes = 1000000, seed = 42):
#     algorithms = ["REINFORCE", "ActorCritic", "A2C", "DQNAgent"]
#     all_results = {}
    
#     for algo in algorithms:
#         all_results[algo] = run_algorithm(algo, num_runs, num_episodes, seed)
        
#     #Plot Comparison
#     plot_learning_curves(all_results, "all_algorithms_comparison.png")
#     plot_comparison_boxplot(all_results, "final_performance_comparison.png")
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description = 'RL Algorithm Comparison')
#     parser.add_argument("--algorithm", type=str, default="all", 
#                     choices=["reinforce", "actor_critic", "a2c", "dqn", "all"],
#                     help="which algorithm to run")
#     parser.add_argument("--runs", type = int, default = 1, help = "Number of runs per algorithm")
#     parser.add_argument("--episodes", type = int, default = 10000, help = "Number of episodes per run")
#     parser.add_argument("--seed", type = int, default = 42, help = "Random seeds")
    
#     args = parser.parse_args()
    
#     # if args.algorithm =="all":
#     #     run_all_algorithms(args.runs, args.episodes, args.seed)
#     # else:
#     #     results = run_algorithm(args.algorithm, args.runs, args.episodes, args.seed)
#     #     plot_learning_curves({args.algorithm: results}, f"{args.algorithm}_learning_curve.png")
#     results = run_algorithm("reinforce", args.runs, args.episodes, args.seed)
#     plot_learning_curves({"reinforce": results}, "reinforce_learning_curve6.png")








# import gymnasium as gym
# import torch
# import random
# import numpy as np
# import argparse
# import os
# from Algorithms.reinforce import REINFORCE
# from Algorithms.actor_critic import ActorCritic
# from Algorithms.a2c import A2C
# from Algorithms.dqn import DQNagent
# # If you want to override the plotting function, you can define it here:
# import matplotlib.pyplot as plt

# def plot_learning_curves(results, filename, smoothing_factor=0.1):
#     """
#     Plots learning curves with Exponential Moving Average (EMA) smoothing.
    
#     :param results: Dictionary where keys are algorithm names and values are lists of runs,
#                     each run being a list of episode rewards.
#     :param filename: The file name for saving the plot.
#     :param smoothing_factor: The EMA smoothing factor (alpha). Typical values are between 0.1 and 0.3.
#     """
#     plt.figure(figsize=(10, 6))
#     for algo, runs in results.items():
#         # Convert list of runs into a NumPy array of shape (num_runs, num_episodes)
#         all_rewards = np.array(runs)
#         # Average across runs
#         avg_rewards = np.mean(all_rewards, axis=0)
#         # Compute EMA
#         ema_rewards = np.zeros_like(avg_rewards)
#         ema_rewards[0] = avg_rewards[0]
#         for t in range(1, len(avg_rewards)):
#             ema_rewards[t] = smoothing_factor * avg_rewards[t] + (1 - smoothing_factor) * ema_rewards[t-1]
#         plt.plot(ema_rewards, label=f"{algo} (EMA α={smoothing_factor})")
    
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.title("Learning Curves (EMA Smoothed)")
#     plt.legend()
#     plt.savefig(filename)
#     plt.close()

# def set_seeds(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
        
# def run_algorithm(algo_name, num_runs=5, num_episodes=500, seed=42):
#     results = []
    
#     # Map input algorithm names to standard formats
#     algo_map = {
#         "reinforce": "REINFORCE",
#         "actor_critic": "ActorCritic",
#         "a2c": "A2C",
#         "dqn": "DQNAgent"
#     }
    
#     # Convert the algorithm name to a standard format
#     standard_name = algo_map.get(algo_name.lower())
#     if standard_name:
#         algo_name = standard_name
    
#     for run in range(num_runs):
#         print(f"\nRunning {algo_name}, Run {run+1}/{num_runs}")
#         set_seeds(seed + run)
#         env = gym.make("CartPole-v1")
        
#         if algo_name == "REINFORCE":
#             agent = REINFORCE(env)  # Uses default hyperparameters from the REINFORCE class.
#         elif algo_name == "ActorCritic":
#             agent = ActorCritic(env, learning_rate=0.001, gamma=0.99)
#         elif algo_name == "A2C":
#             agent = A2C(env, learning_rate=0.001, gamma=0.99)
#         elif algo_name == "DQNAgent":
#             agent = DQNagent(env)
#         else:
#             raise ValueError(f"Unknown Algorithm: {algo_name}")
        
#         rewards = agent.train(num_episodes=num_episodes)
#         results.append(rewards)
#         env.close()
    
#     # Save Results
#     os.makedirs('results', exist_ok=True)
#     np.save(f"results/{algo_name}_rewards.npy", results)
#     return results

# def run_all_algorithms(num_runs=1, num_episodes=100, seed=42):
#     algorithms = ["REINFORCE", "ActorCritic", "A2C", "DQNAgent"]
#     all_results = {}
    
#     for algo in algorithms:
#         all_results[algo] = run_algorithm(algo, num_runs, num_episodes, seed)
        
#     # Plot Comparison
#     plot_learning_curves(all_results, "all_algorithms_comparison.png")
#     # The following function is assumed to exist for boxplot comparisons:
#     from Utils.plotting import plot_comparison_boxplot
#     plot_comparison_boxplot(all_results, "final_performance_comparison.png")
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='RL Algorithm Comparison')
#     parser.add_argument("--algorithm", type=str, default="all", 
#                         choices=["reinforce", "actor_critic", "a2c", "dqn", "all"],
#                         help="which algorithm to run")
#     parser.add_argument("--runs", type=int, default=1, help="Number of runs per algorithm")
#     parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per run")
#     parser.add_argument("--seed", type=int, default=42, help="Random seeds")
    
#     args = parser.parse_args()
    
#     # For running only REINFORCE:
#     results = run_algorithm("reinforce", args.runs, args.episodes, args.seed)
#     plot_learning_curves({"reinforce": results}, "reinforce_learning_curve6.png")
 






import gymnasium as gym
import torch
import random
import numpy as np
import argparse
from Algorithms.reinforce import REINFORCE
from Algorithms.actor_critic import ActorCritic
from Algorithms.a2c import A2C
from Algorithms.dqn import DQNagent
import matplotlib.pyplot as plt

def plot_learning_curves(results, smoothing_factor=0.1):
    """
    Plots learning curves with Exponential Moving Average (EMA) smoothing.
    
    :param results: Dictionary where keys are algorithm names and values are lists of runs,
                    each run being a list of episode rewards.
    :param smoothing_factor: The EMA smoothing factor (alpha). Typical values are between 0.1 and 0.3.
    """
    plt.figure(figsize=(10, 6))
    for algo, runs in results.items():
        # Convert list of runs into a NumPy array of shape (num_runs, num_episodes)
        all_rewards = np.array(runs)
        # Compute the average reward per episode across runs
        avg_rewards = np.mean(all_rewards, axis=0)
        # Compute EMA over the average rewards
        ema_rewards = np.zeros_like(avg_rewards)
        ema_rewards[0] = avg_rewards[0]
        for t in range(1, len(avg_rewards)):
            ema_rewards[t] = smoothing_factor * avg_rewards[t] + (1 - smoothing_factor) * ema_rewards[t-1]
        plt.plot(ema_rewards, label=f"{algo} ")
    
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Learning Curves")
    plt.legend()
    plt.show()  # Show the plot interactively

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def run_algorithm(algo_name, num_runs=1, num_episodes=10000, seed=42):
    results = []
    
    # For this example, we run only REINFORCE
    for run in range(num_runs):
        print(f"\nRunning {algo_name}, Run {run+1}/{num_runs}")
        set_seeds(seed + run)
        env = gym.make("CartPole-v1")
        
        # Instantiate only REINFORCE
        if algo_name.lower() == "reinforce":
            agent = REINFORCE(env)
        else:
            raise ValueError(f"Unknown Algorithm: {algo_name}")
        
        rewards = agent.train(num_episodes=num_episodes)
        results.append(rewards)
        env.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='RL Algorithm Comparison')
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per algorithm")
    parser.add_argument("--episodes", type=int, default=10000
                        , help="Number of episodes per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seeds")
    args = parser.parse_args()
    
    # Run only REINFORCE
    results = run_algorithm("reinforce", args.runs, args.episodes, args.seed)
    plot_learning_curves({"reinforce": results}, smoothing_factor=0.1)

if __name__ == "__main__":
    main()