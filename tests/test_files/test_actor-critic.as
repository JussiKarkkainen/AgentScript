#DEFINE CONFIG
Environment:
  name: CartPole-v1
  horizon: 1000
  preprocess: False

#DEFINE CONFIG
ReplayBuffer:
    episodic: True    
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
    episodes: 5
    batch_size: 1
  meta:
    train: true
    weight_path: None

#DEFINE PYTHON
class Actor:
    def __init__(self, config):
        self.al1 = nn.Linear(4, 128)
        self.al2 = nn.Linear(128, 2)
    
    def __call__(self, state):
        x = self.al2(self.al1(state).relu())
        return x.softmax()

#DEFINE PYTHON
class Critic:
    def __init__(self, config):
        self.cl1 = nn.Linear(4, 128)
        self.cl2 = nn.Linear(128, 1)
    
    def __call__(self, state):
        return self.cl2(self.cl1(state).relu())

#DEFINE PYTHON
def update(networks, replay_buffer, config, environment=None):
    #TODO: Don't love this syntax
    batch = replay_buffer.sample(config["training"]["batch_size"])
    action_prob = networks("Actor", batch["state"])
    action = action_prob.multinomial()
    next_state, reward, terminated, truncated, _ = environment.step(action.item())
    done = terminated or truncated
    
    next_state = Tensor(next_state).unsqueeze(0)       
    replay_buffer.push(next_state)
    reward = Tensor([reward])
    
    state_val = networks("Critic", batch["state"])
    next_state_val = networks("Critic", next_state) if not done else Tensor([0.])
    advantage = reward + config["gamma"] * next_state_val - state_val

    action_one_hot = np.zeros_like(action_prob.squeeze().numpy())
    action_one_hot[action.item()] = 1
    action_one_hot = Tensor(action_one_hot)
    selected_prob = (action_prob.squeeze() * action_one_hot).sum()
    log_prob = selected_prob.log()
   
    actor_loss = (-log_prob * advantage.detach()).sum()
    critic_loss = (((reward + config["gamma"] * next_state_val) - state_val) ** 2).squeeze()
    return (actor_loss, critic_loss), (reward, done)

