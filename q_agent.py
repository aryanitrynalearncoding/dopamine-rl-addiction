# q_agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, intrinsic_beta=0.0):
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = intrinsic_beta
        self.last_delta = 0.0

    def act(self, s, greedy=False):
        # Explore randomly, or exploit the best known action
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next, done):
        # 1. Calculate the target (Reward + discounted future value)
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        
        # 2. Calculate Reward Prediction Error (RPE / Delta)
        delta = target - self.Q[s, a]
        self.last_delta = delta
        
        # 3. Update the Q-table (Dopamine-inspired intrinsic bonus added here)
        learn_target = target + self.beta * abs(delta)
        self.Q[s, a] += self.alpha * (learn_target - self.Q[s, a])
        
        return delta