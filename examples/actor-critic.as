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
    episodes: 1000
    batch_size: 1
  optimizer:
    type: Adam
    learning_rate: 0.001
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, state):
        x = self.layer(state)
        return Categorical(torch.softmax(x, dim=-1))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.layer(state)

#DEFINE PYTHON
def update(network, batch, config):
    action_prob = actor(state)
    action = action_prob.multinomial()
    
    state_val = critic(state)

    next_state_val = critic(next_state) if not done else torch.tensor([0]).float()
    advantage = reward + gamma * next_state_val - state_val

    actor_loss = -action_prob.log_prob(action) * advantage
    critic_loss = F.mse_loss(reward + gamma * next_state_val, state_val)
    return actor_loss, critic_loss



