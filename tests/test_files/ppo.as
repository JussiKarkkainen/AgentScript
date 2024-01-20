#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    update_freq: Episodic 
    type: PPO

#DEFINE CONFIG
Agent:
  type: PPO
  discount_factor: 0.99
  epsilon: 0.2
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
    episodes: 5
    batch_size: 1
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class Policy:
    def __init__(self, config):
        self.l1 = nn.Linear(4, 64)
        self.o = nn.Linear(64, 2)

    def __call__(self, state):
        return self.o(self.l1(state).relu()).softmax()

#DEFINE PYTHON
class Value:
    def __init__(self, config):
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
    
    batch = replay_buffer.sample(config["training"]["batch_size"])
    
    returns = Tensor(compute_returns(batch["rewards"]))
    state_values = networks("Value", batch["states"]).squeeze(1).detach()
    advantages = returns - state_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    log_probs_new = networks("Policy", batch["states"]).log()
    log_probs_new = log_probs_new.gather(batch["actions"].unsqueeze(1), 1)
    state_values = networks("Value", batch["states"]).squeeze(1)
    ratios = (log_probs_new - batch["log_probs_tens"]).exp()
    surr1 = ratios * advantages
    surr2 = Tensor.minimum(Tensor.maximum(ratios, 1-config["epsilon"]), 1+config["epsilon"]) * advantages
    policy_loss = -Tensor.min(surr1, surr2).mean()
    value_loss = (returns - state_values).pow(2).mean()
    return (policy_loss, value_loss), None

