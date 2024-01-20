#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000

#DEFINE CONFIG
ReplayBuffer:
    episodic: True    
    type: PPO

#DEFINE CONFIG
Agent:
  type: PPO
  discount_factor: 0.99
  gamma: 0.99
  update_freq: Episodic
  networks:
    Policy:
      input_shape: 4  
      hidden_layers: [128] 
      output_shape: 2  
    Value:
      input_shape: 4
      hidden_layers: [128]
      output_shape: 1
  optimizer:
    Policy: 
      type: Adam
      learning_rate: 0.001
    Value:
      type: Adam
      learning_rate: 0.001
  training:
    episodes: 1000
    batch_size: 1
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class Policy:
    def __init__(self, n_states, n_actions):
        self.l1 = nn.Linear(4, 64)
        self.o = nn.Linear(64, 2)

    def __call__(self, state):
        return self.o(self.l1(state).relu()).softmax()

#DEFINE PYTHON
class Value:
    def __init__(self, n_states):
        self.l1 = nn.Linear(4, 64)
        self.o = nn.Linear(64, 1)

    def __call__(self, state):
        return self.o(self.l1(state).relu())

#DEFINE PYTHON
def update(networks, replay_buffer, config, environment=None):
    def compute_returns(rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    for _ in range(policy_epochs):
        log_probs_new = (policy_net(states)).log()
            
        log_probs_new = log_probs_new.gather(actions.unsqueeze(1), 1)
        state_values = value_net(states).squeeze(1)

        ratios = (log_probs_new - log_probs_old).exp()

        surr1 = ratios * advantages
        surr2 = Tensor.minimum(Tensor.maximum(ratios, 1-epsilon), 1+epsilon) * advantages
        policy_loss = -Tensor.min(surr1, surr2).mean()

        value_loss = (returns - state_values).pow(2).mean()

    return (policy_loss, value_loss), None

