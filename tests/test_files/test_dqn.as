#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
  capacity: 10000
  batch_size: 64
  episodic: False
  type: DQN

#DEFINE CONFIG
Agent:
  type: DQN
  on_policy: False
  networks:
    DQN:
      input_shape: 4  
      hidden_layers: [128, 512] 
      output_shape: 2  
      activation: relu
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 200
  discount_factor: 0.99
  loss_function: MSELoss
  training:
    episodes: 5
    max_time_steps: 10000
    batch_size: 32
  optimizer:
    DQN:
      type: Adam
      learning_rate: 0.001
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class DQN:
    def __init__(self, config):
        self.layers = []
        input_size = config['input_shape']
        for hidden_size in config['hidden_layers']:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output = nn.Linear(input_size, config['output_shape'])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return self.output(x)

#DEFINE PYTHON
def update(networks, replay_buffer, config, env=None):
    batch = replay_buffer.sample(config["training"]["batch_size"])
    states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
    curr_Q = networks("DQN", states)
    curr_Q = curr_Q.gather(actions.unsqueeze(-1), 1).squeeze(-1)
    next_Q = networks("DQN", next_states).max(1)[0]
    expected_Q = rewards + config["discount_factor"] * next_Q * (1 - dones)
    loss = ((curr_Q - expected_Q.detach()) ** 2).sum()
    return loss, None

