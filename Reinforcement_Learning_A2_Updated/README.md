# RL Algorithms Comparison

This project implements several core reinforcement learning (RL) algorithms using PyTorch and OpenAI Gym environments. 

## 🧠 Implemented Algorithms

- **DQN**: Deep Q-Network with experience replay and a target network.
- **REINFORCE**: Monte Carlo policy gradient method using episodic returns.
- **Actor-Critic (AC)**: Online actor-critic with a shared experience loop.
- **A2C**: Advantage Actor-Critic that uses estimated advantage values.
- **PPO**: Proximal Policy Optimization using clipped surrogate objectives.

All algorithms are implemented modularly and rely on shared network architectures from `networks.py`.
Graphs of all algorithms are plotted using plot_learning_curves function in `plotting.py`

## 🚀 How to Run

### 🔁 Run All Algorithms

You can run all algorithms sequentially using:

```bash
python main.py
