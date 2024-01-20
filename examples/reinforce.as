#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    episodic: True    
    type: REINFORCE

#DEFINE CONFIG
Agent:
  type: REINFORCE
  on_policy: True
  networks:
    Policy:
      input_shape: 4  
      hidden_layers: [128] 
      output_shape: 2  
      activation: relu
  discount_factor: 0.99
  gamma: 0.99
  loss_function: NLLLoss
  training:
    episodes: 1000
    batch_size: 1
  optimizer:
    Policy:
      type: Adam
      learning_rate: 0.001
  meta:
    train: true
    weight_path: None


#DEFINE PYTHON
class Policy:
    def __init__(self, config):
        self.fc = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def __call__(self, x):
        x = self.fc(x).relu()
        x = self.fc2(x)
        return x.softmax().realize()

#DEFINE PYTHON
def update(network, replay_buffer, config, environment=None):
    def calc_qvals(rewards):
        R = 0.0
        qvals = []
        for r in reversed(rewards):
            R = r + config["gamma"] * R
            qvals.insert(0, R)
        return qvals
    
    batch = replay_buffer.sample(config["training"]["batch_size"])
    qvals = calc_qvals(batch["rewards"])
    qvals = Tensor(qvals)
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-5)
    policy_loss = sum([-log_prob * q for log_prob, q in zip(batch["log_probs"], qvals)])
    return policy_loss, None
