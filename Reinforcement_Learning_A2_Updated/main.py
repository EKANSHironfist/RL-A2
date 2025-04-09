# main.py
import gymnasium as gym
import torch
import random
import numpy as np
import argparse
import os
from Algorithms.reinforce import REINFORCE
from Algorithms.actor_critic import ActorCritic
from Algorithms.a2c import A2C
from Algorithms.dqn import DQNagent
from Algorithms.ppo import PPO
from Utils.plotting import plot_learning_curves, plot_comparison_boxplot

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def run_algorithm(algo_name, num_runs=1, max_steps=200000, seed=42):
    results = []
    
    # Map input algorithm names to standard formats
    algo_map = {
        "reinforce": "REINFORCE",
        "actor_critic": "ActorCritic",
        "a2c": "A2C",
        "dqn": "DQNAgent",
        "ppo": "PPO"
    }
    
    # Convert the algorithm name to a standard format
    standard_name = algo_map.get(algo_name.lower())
    if standard_name:
        algo_name = standard_name
    
    for run in range(num_runs):
        print(f"\nRunning {algo_name}, Run {run+1}/{num_runs}")
        set_seeds(seed + run)
        env = gym.make("CartPole-v1")
        
        if algo_name == "REINFORCE":
            agent = REINFORCE(env, learning_rate=0.0005, gamma=0.99)
        elif algo_name == "ActorCritic":
            agent = ActorCritic(env, learning_rate=0.0005, gamma=0.99)
        elif algo_name == "A2C":
            agent = A2C(env, learning_rate=0.0005, gamma=0.99)
        elif algo_name == "DQNAgent":
            agent = DQNagent(env)
        elif algo_name == "PPO":
            agent = PPO(env)
        else:
            raise ValueError(f"Unknown Algorithm: {algo_name}")
        
        step_rewards = agent.train(max_steps=max_steps)  # Now returns list of (step, avg_reward) pairs
        results.append(step_rewards)
        env.close()
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    import pickle
    with open(f"results/{algo_name}_rewards.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_all_algorithms(num_runs=1, max_steps=200000, seed=42):
    algorithms = ["REINFORCE","PPO","ActorCritic","A2C","DQNAgent"]
    all_results = {}
    
    for algo in algorithms:
        all_results[algo] = run_algorithm(algo, num_runs, max_steps, seed)
        
    # Plot Comparison
    plot_learning_curves(all_results, "all_algorithms_comparison.png")
    plot_comparison_boxplot(all_results, "final_performance_comparison.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Algorithm Comparison')
    parser.add_argument("--algorithm", type=str, default="all", 
                        choices=["reinforce", "actor_critic", "a2c", "dqn", "ppo", "all"],
                        help="which algorithm to run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per algorithm")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of environment steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seeds")
    args = parser.parse_args()
    
    if args.algorithm == "all":
        run_all_algorithms(args.runs, args.steps, args.seed)
    else:
        results = run_algorithm(args.algorithm, args.runs, args.steps, args.seed)
        plot_learning_curves({args.algorithm: results}, f"{args.algorithm}_learning_curve.png")
