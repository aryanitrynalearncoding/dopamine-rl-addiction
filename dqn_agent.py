# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.95,
                 epsilon=0.1, intrinsic_beta=0.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = intrinsic_beta
        
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
        # Replay memory
        self.memory = deque(maxlen=2000)
        self.batch_size = 64

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert integer state to one-hot vector
        state_onehot = np.zeros(self.state_size)
        state_onehot[state] = 1
        
        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        # One-hot encode the state for the neural network
        state_onehot = np.zeros(self.state_size)
        state_onehot[state] = 1
        next_state_onehot = np.zeros(self.state_size)
        next_state_onehot[next_state] = 1
        self.memory.append((state_onehot, action, reward, next_state_onehot, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Get next Q values
        with torch.no_grad():
            max_next_q = self.q_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Calculate Delta (RPE)
        delta = target_q - current_q
        
        # Add dopamine intrinsic bonus to the target
        if self.beta > 0:
            target_q = target_q + self.beta * torch.abs(delta)

        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()