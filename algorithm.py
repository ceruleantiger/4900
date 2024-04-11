import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
import random
import torch.nn.functional as F
from run import env


actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.95
tau = 0.01
buffer_size = 5000
batch_size = 128
hidden_size=32
num_agents = 3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        # network layers
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):

        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        #tanh assumes action space between -1 and 1
        #scale it, shift the range from [-1, 1] to [0, 10]
        action = action * 5+5
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_size):
        super(Critic, self).__init__()
        self.input_dim = (state_dim + action_dim) * num_agents

        # Define the network layers
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Outputs a single value (Q-value)

        self.relu = nn.ReLU()

    def forward(self, states, actions):
        # Flatten the states and actions
        x = torch.cat((states.reshape(states.size(0), -1),
                       actions.reshape(actions.size(0), -1)), dim=1)

        # Forward pass through the network with activation functions
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)

        # rewards an agent can obtain by taking a certain action in a given state
        return q_value

# have everything the agent learning from a single timestep
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, batch_size)


class MADDPG:
    def __init__(self, state_dim, action_dim, num_agents, hidden_size, actor_lr, critic_lr, tau):
        self.actors = []
        self.target_actors = []
        self.actor_optimizers = []
        for _ in range(num_agents):
            actor = Actor(state_dim, action_dim, hidden_size)
            self.actors.append(actor)

            target_actor = Actor(state_dim, action_dim, hidden_size)
            self.target_actors.append(target_actor)

            optimizer = Adam(actor.parameters(), lr=actor_lr)
            self.actor_optimizers.append(optimizer)
            

        self.critic = Critic(state_dim * num_agents, action_dim * num_agents, num_agents, hidden_size)
        self.target_critic = Critic(state_dim * num_agents, action_dim * num_agents, num_agents, hidden_size)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau
        self.replay_buffer = ReplayBuffer()
        self.num_agents = num_agents

        # Initialize target networks to match source networks
        self.num_agents = 3
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, obs, i):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        actor = self.actors[i]
        actor.eval()  # set actor network to evaluation mode
        with torch.no_grad():
            action = actor(obs_tensor)
        actor.train()  # set actor network back to train mode

        return action.numpy()

    def update(self, samples):
        obs, actions, rewards, next_obs, dones = samples

        # iterate over each agent
        for i in range(self.num_agents):
            actor = self.actors[i]
            target_actor = self.target_actors[i]
            actor_optimizer = self.actor_optimizers[i]

            # update actor 
            actor_optimizer.zero_grad() # reset gradients of the actor's optimizer
            current_actions = actor(obs)
            # when maximize reward, loss should be minimized
            # so use negative of the Q-values and get every loss by mean
            actor_loss = -self.critic(obs, current_actions).mean()
            # perform backpropagation to compute the gradients of the actor loss
            # update the actor network's weights using the optimizer
            actor_loss.backward()
            actor_optimizer.step()


        # update critic 
        self.critic_optimizer.zero_grad()
        target_actions = []
        for i in range(self.num_agents):
            target_actor = self.target_actors[i]
            next_obs_for_agent = next_obs[i]
            # use target actor network to predict the best action for the next state
            target_action = target_actor(next_obs_for_agent)
            target_actions.append(target_action)
        target_actions = torch.cat(target_actions, dim=1)
        # get Q-values by observation and action
        target_q_values = self.target_critic(next_obs, target_actions).detach()
        # current rewards plus discount factor * future rewards
        # consider whether or not the episode has ended
        # Bellman equation
        expected_q_values = rewards + (gamma * target_q_values * (1 - dones))
        current_q_values = self.critic(obs, actions)
        # loss is the mean squared error between the current and expected Q-values
        critic_loss = F.mse_loss(current_q_values, expected_q_values)
        # Perform backpropagation to compute the gradients of the critic loss
        # update the critic network's weights using its optimizer
        critic_loss.backward()
        self.critic_optimizer.step()


state_dim = 14
action_dim = 2
num_episodes=1000
n_steps=100
num_envs=32
num_agents=3
maddpg = MADDPG(state_dim, action_dim, num_agents, hidden_size, actor_lr, critic_lr, tau)


for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0

    for step in range(n_steps):
        allactions = []
        for i, agent in enumerate(env.agents):
            agent_actions = []
            for env_index in range(num_envs):
                obs = obs[env_index][i]
                action = maddpg.select_action(obs, i)
                agent_actions.append(action)
            
            agent_actions = torch.stack(agent_actions)
            allactions.append(agent_actions)
        
        allactions = torch.stack(allactions, dim=1)

        next_obs, rewards, dones, info = env.step(allactions)
        maddpg.replay_buffer.add((obs, allactions, rewards, next_obs, dones))

        total_reward += sum(rewards)
        obs = next_obs

        if any(dones):
            break

        if len(maddpg.replay_buffer) > batch_size:
            samples = maddpg.replay_buffer.sample()
            maddpg.update(samples)

print(f'Episode {episode} Total Reward: {total_reward}')


