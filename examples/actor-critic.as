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
  loss_function: MSELoss
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
class Actor:
    def __init__(self):
        self.al1 = nn.Linear(4, 128)
        self.al2 = nn.Linear(128, 2)
    
    def __call__(self, state):
        x = self.al2(self.al1(state).relu())
        return x.softmax()

class Critic:
    def __init__(self):
        self.cl1 = nn.Linear(4, 128)
        self.cl2 = nn.Linear(128, 1)
    
    def __call__(self, state):
        return self.cl2(self.cl1(state).relu())

#DEFINE PYTHON
def update(network, batch, config):
    action_prob = actor(state)
    action = action_prob.multinomial()
    
    next_state, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated
    
    next_state = Tensor(next_state).unsqueeze(0)       
    reward = Tensor([reward])
    rewards.append(reward)
    
    state_val = critic(state)
    next_state_val = critic(next_state) if not done else Tensor([0.])
    advantage = reward + gamma * next_state_val - state_val

    action_one_hot = np.zeros_like(action_prob.squeeze().numpy())
    action_one_hot[action.item()] = 1
    action_one_hot = Tensor(action_one_hot)
    selected_prob = (action_prob.squeeze() * action_one_hot).sum()
    log_prob = selected_prob.log()
   
    actor_loss = (-log_prob * advantage.detach()).sum()
    critic_loss = (((reward + gamma * next_state_val) - state_val) ** 2).squeeze()
    return actor_loss, critic_loss



