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
  loss_function: NLLLoss
  training:
    episodes: 1000
    batch_size: None
  optimizer:
    type: Adam
    learning_rate: 0.001
  meta:
    train: true
    weight_path: None


#DEFINE PYTHON
class Network:
    def __init__(self, config):
        input_size = config['network']['input_shape']
        for hidden_size in config["network"]["hidden_layers"]:
            self.fc = nn.Linear(input_size, hidden_size)
            input_size = hidden_size
        self.fc2 = nn.Linear(input_size, config["network"]["output_shape"])

    def __call__(self, x):
        x = self.fc(x).relu()
        x = self.fc2(x)
        return x.softmax(axis=1)


#DEFINE PYTHON
def update(network, batch, config):
    def calc_qvals(rewards):
        R = 0.0
        qvals = []
        for r in reversed(rewards):
            R = r + gamma * R
            qvals.insert(0, R)
        return qvals
    
    qvals = calc_qvals(batch["rewards"])
    qvals = Tensor(qvals)
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-5)
    policy_loss = sum([-log_prob * q for log_prob, q in zip(batch["log_probs"], qvals)])
    return policy_loss
