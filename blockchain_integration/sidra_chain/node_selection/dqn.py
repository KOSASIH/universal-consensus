# dqn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym
from gym import spaces
import random

# Define the Node Selection Environment
class NodeSelectionEnvironment(gym.Env):
    def __init__(self, num_nodes, num_actions):
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.state_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)
        self.current_state = np.zeros((num_nodes,))
        self.current_action = None
        self.reward = 0
        self.done = False

    def reset(self):
        self.current_state = np.zeros((self.num_nodes,))
        self.current_action = None
        self.reward = 0
        self.done = False
        return self.current_state

    def step(self, action):
        self.current_action = action
        self.reward = self.calculate_reward()
        self.done = self.is_done()
        self.current_state = self.next_state()
        return self.current_state, self.reward, self.done, {}

    def calculate_reward(self):
        # Calculate the reward based on the current state and action
        # For example, reward = -1 if the action is invalid, reward = 1 if the action is valid
        reward = -1 if self.current_action >= self.num_actions else 1
        return reward

    def is_done(self):
        # Check if the episode is done
        # For example, done = True if the current state is a terminal state
        done = np.all(self.current_state == 1)
        return done

    def next_state(self):
        # Calculate the next state based on the current state and action
        # For example, next_state = current_state + action
        next_state = self.current_state.copy()
        next_state[self.current_action] = 1
        return next_state

# Define the Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Node Selection Agent
class NodeSelectionAgent:
    def __init__(self, num_nodes, num_actions):
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.dqn = DQN(num_nodes, num_actions)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
        batch = random.sample(self.memory, 32)
        states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        q_values = self.dqn(states)
        next_q_values = self.dqn(next_states)
        q_targets = rewards + self.gamma * next_q_values.max(dim=1)[0] * (1 - dones)
        loss = (q_values[range(32), actions] - q_targets).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

# Train
    def train(self, num_episodes):
        writer = SummaryWriter()
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            rewards = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                rewards += reward
            writer.add_scalar('Episode Reward', rewards, episode)
            print(f'Episode {episode+1}, Reward: {rewards}')
        writer.close()

    def test(self, num_episodes):
        rewards = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)
                rewards += reward
        print(f'Test Reward: {rewards / num_episodes}')

# Create the environment and agent
env = NodeSelectionEnvironment(num_nodes=10, num_actions=5)
agent = NodeSelectionAgent(num_nodes=10, num_actions=5)

# Train the agent
agent.train(num_episodes=1000)

# Test the agent
agent.test(num_episodes=100)
