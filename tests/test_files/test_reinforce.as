#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    episodic: True    

#DEFINE CONFIG
Agent:
  type: REINFORCE
  on_policy: True
  network:
    input_shape: 4  
    hidden_layers: [128] 
    output_shape: 2  
    activation: relu
  discount_factor: 0.99
  gamma: 0.99
  loss_function: NLLLoss
  training:
    episodes: 5
    batch_size: 1
  optimizer:
    type: Adam
    learning_rate: 0.001
  meta:
    train: true
    weight_path: None


#DEFINE PYTHON
class Network:
    def __init__(self, config):
        self.fc = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def __call__(self, x):
        x = self.fc(x).relu()
        x = self.fc2(x)
        return x.softmax().realize()

#DEFINE PYTHON
def update(network, batch, config):
    def calc_qvals(rewards):
        R = 0.0
        qvals = []
        for r in reversed(rewards):
            R = r + config["gamma"] * R
            qvals.insert(0, R)
        return qvals
    
    qvals = calc_qvals(batch["rewards"])
    qvals = Tensor(qvals)
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-5)
    policy_loss = sum([-log_prob * q for log_prob, q in zip(batch["log_probs"], qvals)])
    return policy_loss
