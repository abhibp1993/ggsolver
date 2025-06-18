import random
import pickle 
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space, flatten

from automatica2025 import *
from pathlib import Path


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

        # print(set(state.aut_state for state in self._game.states()
        # ))
        # key=input("stop here")
        # print(self._game.aut.pref_graph)
        # print(type(self._game.aut.pref_graph))
        # print(self.find_nodes_without_outgoing_edges(self._game.aut.pref_graph))
        
        # dictionary of node ranks {node: rank, node: rank, ...}
        self.ranks, self.node_partitions = self.find_nodes_without_outgoing_edges(self._game.aut.pref_graph)
        print("Node ranks:", self.ranks)
        print("Node partitions:", self.node_partitions)
        #key=input("stop here")
        #keeps track of preference aut states visited 
        self.pref_aut = []
        self.ranks_visited = []

        self._grid_rows = config["num_rows"] # Y VALUES
        self._grid_cols = config["num_columns"] # X VALUES
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
        
        #[bee_x, bee_y, bird_x, bird_y, battery, raining, rain_prob, terminated, aut_state]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0.0, 0, 0], dtype=np.float32),
            high=np.array([self._grid_cols-1, self._grid_rows-1, self._grid_cols-1, self._grid_rows-1, self._battery_capacity, 1, 1.0, 1, self._game.model.number_of_nodes()], dtype=np.float32),
        )
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

    def find_nodes_without_outgoing_edges(self, graph):
        """
            Iteratively finds nodes without outgoing edges, removes them from the graph,
            assigns ranks to the nodes, and returns a dictionary of node ranks.

            :param graph: A networkx graph (e.g., nx.MultiDiGraph).
            :return: A dictionary mapping nodes to their ranks.
        """
        remaining_graph = graph.copy()  # Create a copy of the graph to modify
        node_ranks = {}  # Dictionary to store the rank of each node
        # {node: rank, node: rank, ...}
        current_rank = 0  # Start with rank 0 (highest preference)

        print(remaining_graph.number_of_nodes(), "nodes in the graph")
        print(remaining_graph.nodes(data=True))
        #{key is pref_node number, value is semi-aut nodes}
        node_to_partition = {node: data['partition'] for node, data in remaining_graph.nodes(data=True)}
        print("Node to partition mapping:", node_to_partition)
        print(remaining_graph.number_of_edges(), "edges in the graph")
        print(remaining_graph.edges)

        while remaining_graph.number_of_nodes() > 0:
            # Find nodes without outgoing edges, excluding self-loops
            nodes_without_outgoing = [
                node for node in remaining_graph.nodes
                if remaining_graph.out_degree(node) == 0 or
                all(v == node for _, v, _ in remaining_graph.out_edges(node, keys=True))
            ]
            #print(f"Found {len(nodes_without_outgoing)} nodes without outgoing edges: {nodes_without_outgoing}")

            # If no nodes without outgoing edges are found, break the loop
            if not nodes_without_outgoing:
                break  # Stop if no nodes without outgoing edges are found

            # Assign the current rank to these nodes
            for node in nodes_without_outgoing:
                node_ranks[node] = current_rank

            # Remove these nodes from the graph
            remaining_graph.remove_nodes_from(nodes_without_outgoing)

            # Increment the rank for the next iteration
            current_rank += 1

        #print("Node ranks:", node_ranks)
        return node_ranks, node_to_partition  # Return the dictionary of node ranks and the mapping of nodes to partitions
    
    def convert(self, state):
        """
            Takes ProdState(game_state=MDPState(bee_x=1, bee_y=0, bird_x=3, bird_y=1, battery=12, raining=False, rain_prob=0.2, terminated=False), aut_state=3)
            and converts to observation_space array

        """
        # if terminated, return a zeroed observation space
        if(state.game_state.terminated):
            return np.array([0, 0, 0, 0, 0, 0, 0.0, 1, state.aut_state], dtype=np.float32)

        obs_space = np.array([float(state.game_state.bee_x), float(state.game_state.bee_y), float(state.game_state.bird_x), float(state.game_state.bird_y), float(state.game_state.battery), 
                    float(state.game_state.raining), float(state.game_state.rain_prob), float(state.game_state.terminated), float(state.aut_state)], dtype=np.float32)

        return obs_space

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed) 
        self.state = set(self._game.model.graph["init_states"]).pop() # needed for internal model transition
        obs_state = self._game.model.nodes[self.state]["state"]
        self.obs_space = self.convert(obs_state) # needed for RL algorithm 
        
        return self.obs_space, {}
    
    def give_reward(self, current_semi_aut_state, next_semi_aut_state):
        """
        Calculate the reward based on the ranking of the current and next automaton states.
        If the rank decreases (i.e., preference improves), a higher reward is given.

        :param current_aut_state: The current automaton state.
        :param next_aut_state: The next automaton state.
        :return: A reward value.
        """
        print(f"Current semi automaton state: {current_semi_aut_state}, Next semi automaton state: {next_semi_aut_state}")
        print(type(self.node_partitions))
        print(self.node_partitions)
        current_pref_state = [k for k, v in self.node_partitions.items() if current_semi_aut_state in v][0]
        next_pref_state = [k for k, v in self.node_partitions.items() if next_semi_aut_state in v][0]

        print(f"Current preference state: {current_pref_state}, Next preference state: {next_pref_state}")

        current_rank = self.ranks.get(current_pref_state, float('inf'))  # Default to infinity if state is not ranked
        next_rank = self.ranks.get(next_pref_state, float('inf'))  # Default to infinity if state is not ranked

        print(f"Current rank: {current_rank}, Next rank: {next_rank}")
        self.pref_aut.append(current_pref_state)  # Keep track of the preference automaton states visited
        self.ranks_visited.append(current_rank)  # Keep track of the ranks visited

        if next_rank < current_rank:
            # Preference improves (rank decreases)
            reward = (current_rank - next_rank)*100  # Reward is proportional to the rank improvement
            # print(f"Reward for improving preference: {reward}")
        # elif next_rank == current_rank:
        #     # Preference remains the same
        #     reward = 0
        #     print("No change in preference, reward is 0")
        #     print(f"Reward for no change in preference: {reward}")
        else:
            # Preference does not improve or worsens
            reward = -1  # Penalize for worsening or no improvement
            # print("Preference worsens or no improvement, reward is -1")
            # print(f"Reward for worsening preference: {reward}")

        return reward


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
        current_aut_state = current_state.aut_state
        # print("current state is ")
        # print(current_state)
        # print("current aut state is ")
        # print(current_aut_state)
        next_states = self._game.delta(current_state, action_name)
        # print("next states are ")
        # print(next_states)

        if not next_states:
            #print("No next states available, Terminating")
            terminated = True
            truncated = True
            reward = -1  # Penalize for no available next states
            info = {"state": None}
            return self.obs_space, reward, terminated, truncated, info

        next_state = random.sample(sorted(next_states), 1).pop()
        # print("chosen next state is ")
        # print(next_state)
        next_aut_state = next_state.aut_state
        # print("next aut state is ")
        # print(next_aut_state)
        reward = self.give_reward(current_aut_state, next_aut_state)
        # print("reward is ")
        # print(reward)

        # Update the environment's state
        self.state = self._state2id[next_state]
        self.obs_space = self.convert(next_state)
        terminated = next_state.game_state.terminated
        truncated = terminated
        
        info = {"state": next_state}
        
        # if self.render_mode == "human":
        #     self._render_frame()

        # Return the new state, reward, done flag, and additional info
        #return self.state, 0, None, False, info
        return self.obs_space, reward, terminated, truncated, info

    def render(self):
        """
        Render the current state of the environment.
        """
        return self._render_frame()

    def _render_frame(self):
        pass


if __name__ == '__main__':
    with open(Path().absolute().parent /"automatica2025" /".tmp" / "model.pkl", "rb") as model_file:
        prod_game = pickle.load(model_file)

    with open(Path().absolute().parent /"automatica2025" / ".tmp" / "solutions.pkl", "rb") as model_file:
        solutions = pickle.load(model_file)
        solver = solutions[0]
    
    CONFIG = {
        "num_columns": 5,
        "num_rows": 4,
        "actions": ["N", "E", "S", "W", "Y", "T"],
        "bee_initial_loc": (1, 0),
        "bird_initial_loc": (3, 1),
        "battery_capacity": 12,
        "bird_bounds": {(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)},
        "tulip_loc": (4, 3),
        "orchid_loc": (1, 1),
        "daisy_loc": (0, 2),
        "bee_dynamic_stochastic": False,
        "bee_dynamic_stochasticity_prob": 0.1,
        "spec_file_path": Path().parent.absolute() / "beerobot.prefltlf"
    }
    env = BeeRobotEnv(
        config=CONFIG,
        game=prod_game,
        solver=solver,
        render_mode="human",
    )
    num_episodes = 2000
    max_timesteps = 500
    aut_states = []
    for ep in range(num_episodes):
        state, info = env.reset()
        print(f"\n=== EPISODE {ep+1} ===")
        for t in range(max_timesteps):
            print(f"=====================Step {t+1} in Episode {ep+1}")
            act = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(act)
            # Extract aut_state from info or next_state
            aut_state = None
            if "state" in info and info["state"] is not None:
                aut_state = info["state"].aut_state
                aut_states.append(aut_state)
            print(f"Step {t+1}: Action: {act}, aut_state: {aut_state}, Reward: {reward}, Terminated: {terminated}")
            print(info)
            state = next_state
            if terminated:
                print("Episode terminated.")
                print("uniqiue automaton states encountered in this episode:")
                print(set(aut_states))  # Print unique automaton states encountered in this episode
                #aut_states = []  # Reset for the next episode
                break
    print("All unique pref automaton states encountered across all episodes:")
    print(set(env.pref_aut))  # Print unique automaton states encountered across all episodes
    print("all ranks visited ")
    print(set(env.ranks_visited))
    # state, info = env.reset()
    # print(type(state))
    # print("obs space below")
    # print(state, info)  # -> int (representing node id), dict
    # print(env.action_space)
    # for i in range(15):
    #     print("EPISODE " + str(i)) 
    #     act = env.action_space.sample()  # -> int
    #     state, reward, terminated, truncated, info = env.step(act)
    #     print(f"Action: {act}, State: {state}, Reward: {reward}, Terminated: {terminated}, info: {info}")


## install MONA on system 


