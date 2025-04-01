Initialize:
    policy_net ← PolicyNetwork(), value_net ← ValueNetwork()
    policy_optimizer, value_optimizer ← Adam optimizers
    
Loop forever (for each episode):
    S₀ ← reset environment
    done ← False
    While not done:
        Aₜ ← sample action from policy_net(Sₜ)
        Execute action Aₜ, observe reward Rₜ₊₁ and next state Sₜ₊₁
        TD_target ← Rₜ₊₁ + γ * value_net(Sₜ₊₁)
        TD_error ← TD_target - value_net(Sₜ)
        
        value_loss ← (TD_error)²
        policy_loss ← - log π(Aₜ|Sₜ) * TD_error (detach TD_error)
        
        Perform gradient descent on value_net using value_loss
        Perform gradient ascent on policy_net using policy_loss
        
        Sₜ ← Sₜ₊₁