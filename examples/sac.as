#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    update_freq: Timestep
    type: ActorCritic

#DEFINE CONFIG
Agent:
  type: ActorCritic
  discount_factor: 0.99
  gamma: 0.99
  update_freq: Timestep
  networks:
    Actor:
      input_shape: 4  
      hidden_layers: [128] 
      output_shape: 2  
      activation: relu
    Critic:
      input_shape: 4
      hidden_layers: [128]
      output_shape: 1
      activation: relu
  loss_function: MSELoss
  optimizer:
    Actor: 
      type: Adam
      learning_rate: 0.001
    Critic:
      type: Adam
      learning_rate: 0.001
  training:
    episodes: 1000
    batch_size: 1
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fc1(state).relu()
        x = self.fc2(x).relu()
        action = self.fc3(x).tanh()
        return action

#DEFINE PYTHON
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = self.fc1([state, action].cat(), 1)).relu()
        x = self.fc2(x).relu()
        value = self.fc3(x)
        return value

#DEFINE PYTHON
def update(networks, replay_buffer, config, environment=None):
    pass


