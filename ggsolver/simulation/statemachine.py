import json
import random
from abc import abstractmethod
from typing import Dict, Callable, Any, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from loguru import logger as default_logger


class Simulator(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
            self,
            game,
            monitors: Dict[str, Callable] = None,
            logger=default_logger,
            render_mode=None,
            *args,
            **kwargs
    ):
        super(Simulator, self).__init__()
        self.game = game
        self.monitors = monitors or {}
        self.logger = logger

        # State tracking
        self.state = None
        self.history: List[Tuple[Any, Any, Any]] = []
        self.future: List[Tuple[Any, Any, Any]] = []

        # Random generator
        self.seed_value = kwargs.get("seed", None)
        self.rng = random.Random(self.seed_value)

        # Observation and action space placeholders
        self.observation_space = spaces.Discrete(1)  # Should be customized later
        self.action_space = spaces.Discrete(1)  # Should be customized later

        # Init state
        self.reset()

        # Pygame window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        init_states = list(self.game.init_states())
        self.state = self.rng.choice(init_states)
        obs = self._get_obs()
        info = self._get_info()
        self.history = [(self.state, None, obs, info)]
        self.future.clear()
        return obs, info

    @abstractmethod
    def delta(self, state, action):
        pass

    def step(self, action, *, clear_future=False):
        if not clear_future and self.future:
            self.state, action, obs = self.future.pop(0)
        else:
            next_state = self.delta(self.state, action)
            self.state = next_state
            obs = self._get_obs()
            info = self._get_info()
            self.history.append((self.state, action, obs, info))

            # Check monitors
            for name, monitor_fn in self.monitors.items():
                if not monitor_fn(self.history):
                    self.logger.warning(f"Monitor '{name}' triggered on history: {self.history}")

            # Logging
            self.logger.info(f"Step: {len(self.history)} | Action: {action} | State: {self.state} | Obs: {obs}")

        info = self._get_info()
        done = self._is_done()
        reward = self._get_reward()
        return obs, reward, done, False, info

    def back(self):
        if self.history:
            self.future.insert(0, (self.state, *self.history[-1][1:]))
            self.state, _, _ = self.history.pop()
        return self._get_obs(), self._get_info()

    def run(self, policies: Dict[int, Callable[[Any], Any]], steps: Optional[int] = None):
        count = 0
        while steps is None or count < steps:
            current_turn = self.game.turn(self.state)
            if current_turn not in policies:
                raise ValueError(f"No policy provided for player {current_turn}")
            action = policies[current_turn](self.state)
            self.step(action)
            count += 1

    def seed(self, seed=None):
        self.seed_value = seed
        self.rng.seed(seed)

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump([
                {"state": str(s), "action": a, "obs": str(o)}
                for s, a, o in self.history
            ], f, indent=2)

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return dict()

    def _is_done(self):
        return False

    def _get_reward(self):
        rewards = self.game.reward(self.state)
        return sum(rewards.values()) if isinstance(rewards, dict) else 0
