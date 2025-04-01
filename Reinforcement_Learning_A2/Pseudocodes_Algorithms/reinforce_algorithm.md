Initialize:
    policy_net ← PolicyNetwork(), optimizer ← Adam(policy_net.parameters())
    
Loop forever (for each episode):
    Generate an episode trajectory: (S₀, A₀, R₁, ..., Sₜ₋₁, Aₜ₋₁, Rₜ)
    G ← 0
    For each step t of episode, t = T-1, T-2, …, 0:
        G ← γG + Rₜ₊₁
        Compute log probability: log π(Aₜ|Sₜ)
        Compute returns: Returns[t] ← G
    Normalize Returns
    policy_loss ← - ∑ (log π(Aₜ|Sₜ) * Returns[t])  
    Perform gradient ascent on policy_net using policy_loss