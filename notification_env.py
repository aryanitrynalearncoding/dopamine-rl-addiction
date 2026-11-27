# notification_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NotificationEnv(gym.Env):
    """Minimal smartphone-notification environment.

    State : steps_since_last_reward, capped at N_STATES-1
    Actions: 0 = task (+5 fixed), 1 = notify (+20 w.p. p_notif), 2 = idle (0)
    """

    N_STATES = 5
    HORIZON = 100
    TASK_REWARD = 5.0
    NOTIF_REWARD = 20.0

    def __init__(self, p_notif=0.1, notify_enabled=True, seed=None):
        super().__init__()
        self.p_notif = p_notif
        self.notify_enabled = notify_enabled
        self.observation_space = spaces.Discrete(self.N_STATES)
        self.action_space = spaces.Discrete(3)
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._since = 0

    def _obs(self):
        return min(self._since, self.N_STATES - 1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._since = 0
        return self._obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        if action == 0:                      # do task
            reward = self.TASK_REWARD
            self._since += 1
        elif action == 1:                    # check notification
            if self.notify_enabled and self.rng.random() < self.p_notif:
                reward = self.NOTIF_REWARD
                self._since = 0
            else:
                self._since += 1
        else:                                # idle
            self._since += 1

        self._t += 1
        terminated = False
        truncated = self._t >= self.HORIZON
        return self._obs(), reward, terminated, truncated, {}