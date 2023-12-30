#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
  capacity: 10000
  batch_size: 64

#DEFINE CONFIG
Agent:
  type: DQN
  network:
    input_shape: 4  
    hidden_layers: [32, 64] 
    output_shape: 2  
    activation: relu
  exploration:
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 200
  memory: replay_buffer 
  tnameget_update_frequency: 100
  discount_factor: 0.99
  loss_function: huber_loss
  training:
    episodes: 1000
    max_time_steps: 10000
  optimizer:
    type: Adam
    learning_rate: 0.001
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class DQNNetwork(nn.Module):
    def __init__(self, config):
        super(DQNNetwork, self).__init__()
        self.layers = nn.ModuleList()
        input_size = config['network']['input_shape']
        for hidden_size in config['network']['hidden_layers']:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output = nn.Linear(input_size, config['network']['output_shape'])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

#DEFINE PYTHON
def dqn_update(agent, batch):
    states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
    curr_Q = agent.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_Q = agent.target_network(next_states).max(1)[0]
    expected_Q = rewards + (agent.gamma * next_Q * (1 - dones))
    loss = F.mse_loss(curr_Q, expected_Q.detach())
    return loss

