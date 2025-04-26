import random

import gymnasium as gym

from automatica2025 import *


class BeeRobotEnv(gym.Env):
    """
    Bee robot environment.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
            self,
            config: dict,
            game: PrefGraphGame,
            solver: Solver,
            render_mode=None,
            **kwargs
    ):
        super(BeeRobotEnv, self).__init__()

        self._config = config
        self._game = game
        self._solver = solver
        self._policy = solver._policy

        self._grid_rows = config["num_rows"]
        self._grid_cols = config["num_columns"]
        self._obstacles = set(tuple(obs) for obs in config.get("obstacles", []))
        self._actions = list(config["actions"])

        self._bee_initial_loc = config["bee_initial_loc"]
        self._bird_initial_loc = config["bird_initial_loc"]
        self._battery_capacity = config["battery_capacity"]
        self._bird_bounds = set(tuple(x) for x in config["bird_bounds"])

        self._tulip_loc = config["tulip_loc"]
        self._orchid_loc = config["orchid_loc"]
        self._daisy_loc = config["daisy_loc"]

        self._bee_dynamic_stochastic = config["bee_dynamic_stochastic"]
        self._bee_dynamic_stochasticity_prob = config["bee_dynamic_stochasticity_prob"]

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self._actions))  # Actions are indexed
        self._act2id = {action: act_id for act_id, action in enumerate(self._actions)}
        self.observation_space = gym.spaces.Discrete(self._game.model.number_of_nodes())
        self._obs2id = self._state2id = {data["state"]: node for node, data in self._game.model.nodes(data=True)}

        # Initialize pygame params
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.size = max(self._grid_rows, self._grid_cols)

        # Initialize state
        self.state = set(self._game.model.graph["init_states"]).pop()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.state = set(self._game.model.graph["init_states"]).pop()
        return self.state, {}

    def step(self, action):
        """
        Execute an action in the environment.
        """
        assert self.action_space.contains(action), f"Invalid action:{action}"

        # Map action index to action name
        action_name = self._actions[action]

        # Compute the next state using the model's delta function
        current_node = self.state
        current_state = self._game.model.nodes[current_node]["state"]
        next_states = self._game.delta(current_state, action_name)
        next_state = random.sample(sorted(next_states), 1).pop()

        # Update the environment's state
        self.state = self._state2id[next_state]
        info = {"state": next_state}

        # if self.render_mode == "human":
        #     self._render_frame()

        # Return the new state, reward, done flag, and additional info
        return self.state, 0, None, False, info

    def render(self):
        """
        Render the current state of the environment.
        """
        return self._render_frame()

    def _render_frame(self):
        pass


if __name__ == '__main__':
    with open(Path().absolute().parent / ".tmp" / "model.pkl", "rb") as model_file:
        prod_game = pickle.load(model_file)

    with open(Path().absolute().parent / ".tmp" / "solutions.pkl", "rb") as model_file:
        solutions = pickle.load(model_file)
        solver = solutions[0]

    env = BeeRobotEnv(
        config=CONFIG,
        game=prod_game,
        solver=solver,
        render_mode="human",
    )
    observation, info = env.reset()
    print(observation, info)  # -> int (representing node id), dict
    print(act := env.action_space.sample())  # -> int
    env.step(act)
