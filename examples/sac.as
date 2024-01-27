#DEFINE CONFIG
Environment:
  name: Pendulum-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    update_freq: Batch
    type: SAC

#DEFINE CONFIG
Agent:
  type: SAC
  discount_factor: 0.99
  gamma: 0.99
  update_freq: Batch
  networks:
    Actor
    critic1
    critic2
    critic1_target
    critic2_target
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
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


#DEFINE PYTHON
def update(networks, replay_buffer, config, environment=None):
    state, next_state, action, reward, done = replay_buffer.sample(config["batch_size"])

    # with no_grad() equivelant
    Tensor.no_grad = True
    next_action = networks("Actor", next_state)
    next_state_action_values1 = networks("critic1_target", (next_state, next_action))
    next_state_action_values2 = networks("critic2_target", (next_state, next_action))
    next_state_action_values = Tensor.minimum(next_state_action_values1, next_state_action_values2)
    target_Q = reward + (1 - done) * gamma * next_state_action_values
    critic1_loss = (networks("critic1", (state, action)) - target_Q) ** 2
    Tensor.no_grad = False
    
    critic1_loss = (neteorks("critic1", (state, action)) - target_Q) ** 2

    critic2_loss = (networks("critic2", (state, action)) - target_Q) ** 2

    actor_loss = -networks("critic1", (state, networks("Actor", (state)))).mean()

    soft_update(networks["critic1_target"], networks["critic1"], config["tau"])
    soft_update(networks["critic2_target"], networks["critic2"], config["tau"])


