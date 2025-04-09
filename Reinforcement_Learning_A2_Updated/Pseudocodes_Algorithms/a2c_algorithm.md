Initialize:
    policy_net ← PolicyNetwork(), value_net ← ValueNetwork()
    policy_optimizer, value_optimizer ← Adam optimizers

Loop forever (for each episode):
    Generate an episode trajectory: (S₀, A₀, R₁, …, Sₜ₋₁, Aₜ₋₁, Rₜ)
    Compute discounted returns: Returns[t] for all t
    Compute values: Vₜ = value_net(Sₜ) for all t
    Compute advantages: Aₜ ← Returns[t] - Vₜ
    policy_loss ← - mean(log π(Aₜ|Sₜ) * Aₜ)
    value_loss ← mean((Returns[t] - Vₜ)²)

    Perform gradient ascent on policy_net using policy_loss
    Perform gradient descent on value_net using value_loss