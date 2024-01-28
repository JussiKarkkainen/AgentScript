#DEFINE CONFIG
Environment:
  name: CartPole-v1
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
  capacity: 10000
  batch_size: 32
  update_freq: Batch
  type: DQN

#DEFINE CONFIG
Agent:
  type: DQN
  update_freq: Batch
  networks:
    DQN:
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 200
  discount_factor: 0.99
  training:
    episodes: 230
    max_time_steps: 10000
    batch_size: 32
  optimizer:
    DQN:
      type: Adam
      learning_rate: 0.001
  meta:
    train: True
    weight_path: "weights/dqn"
  logs: 
    logging: True
    config: [exploration, discount_factor, training, optimizer]

#DEFINE NN
class DQN:
    def __init__(self, config):
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 2)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)

#DEFINE PYTHON
def update(networks, replay_buffer, config, env=None):
    batch = replay_buffer.sample(config["training"]["batch_size"])
    states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
    curr_Q = networks("DQN", states)
    curr_Q = curr_Q.gather(actions.unsqueeze(-1), 1).squeeze(-1)
    next_Q = networks("DQN", next_states).max(1)
    expected_Q = rewards + config["discount_factor"] * next_Q * (1 - dones)
    loss = Tensor.mean((curr_Q - expected_Q.detach()) ** 2)
    return loss, None

