import json
import itertools
from ggsolver.decoy_alloc.models import ReachabilityGame
import networkx as nx
from networkx.readwrite import json_graph

class CBSGame(ReachabilityGame):
    """
    CyberBattleSim game
    """
    def __init__(self, json_fname, final_node):
        super(ReachabilityGame, self).__init__()
        self._init_state = ("0", (0, 0, 0), (1, 1, 1, 1, 1), 2, (1, 0, 0, 0, 0, 0))

        self._json_fname = json_fname
        self._final_node = final_node
        self._cbs_network = self._decode()
        self._states, self._credential_set, self._possible_firewall_states, self._connections, self._final_states = self._construct_states()
        self._actions = self._construct_actions()

        self._ATTACKER_TURN = 2
        self._DEFENDER_TURN = 1

    def _decode(self):
        # do something with self._json
        # return nx.MultiDiGraph()
        with open(self._json_fname) as f:
            data = json.load(f)
        network = json_graph.adjacency_graph(data, directed=True, multigraph=True)
        return network

    def states(self):
        return self._states

    def actions(self):
        return self._actions

    def delta(self, state, act):
        ### state is of form (int source_node, bool[] obtained_credentials, bool[] firewall_state, int turn, bool[] owned nodes) ###
        source_node = state[0]
        obtained_credentials = state[1]
        firewall_state = state[2]
        turn = state[3]
        owned_nodes = state[4]

        ## attacker actions ##
        if turn == self._ATTACKER_TURN:
            no_attacker_action_state = (source_node, obtained_credentials, firewall_state, self._DEFENDER_TURN, owned_nodes)

            if act == "no_attacker_action":
                return no_attacker_action_state
            elif act[0:13] == "move_to_node_":
                target_node = act[13:]
                if(owned_nodes[int(target_node)]) == 1:
                    return (target_node, obtained_credentials, firewall_state, self._DEFENDER_TURN, owned_nodes)
                else:
                    return no_attacker_action_state
            elif act[0:16] == "local_attack_on_":
                target_node = act[16:]
                if target_node == source_node:
                    # find new credentials
                    new_obtained_credentials = list(obtained_credentials)
                    for credential in self._cbs_network.nodes[target_node]["creds_stored"]:
                        # the "creds_stored" is a list of credentials of form (node_to_be_used_on, service, credential)
                        if new_obtained_credentials[int(credential[2])] == 0:
                            new_obtained_credentials[int(credential[2])] = 1
                    return (source_node, tuple(new_obtained_credentials), firewall_state, self._DEFENDER_TURN, owned_nodes)
                else:
                    return no_attacker_action_state
            elif act[0:11] == "connect_to_":
                index = act.find("_with_")
                target_node = act[11:index]
                credential = act[index+6:]

                # Check conditions for connection
                source_connected_to_target = False
                connection_index = 0
                for i, connection in enumerate(self._connections):
                    if connection == (source_node, target_node):
                        source_connected_to_target = True
                        connection_index = i
                target_accepts_credential = credential in self._cbs_network.nodes[target_node]["allowed_creds"]
                connection_allowed_by_firewall = bool(firewall_state[connection_index])

                if source_connected_to_target and target_accepts_credential and connection_allowed_by_firewall:
                    # return state that is the same as passed state but at new node
                    new_owned_nodes = list(owned_nodes)
                    new_owned_nodes[int(target_node)] = 1
                    return (target_node, obtained_credentials, firewall_state, self._DEFENDER_TURN, tuple(new_owned_nodes))
                else:
                    return no_attacker_action_state
            else:
                return no_attacker_action_state
        ## defender action ##
        else:
            if act[0:19] == "change_firewall_to_":
                new_firewall_state_string = act[19:]
                new_firewall_state = [int(i) for i in new_firewall_state_string.strip(')(').split(', ')]
                return (source_node, obtained_credentials, tuple(new_firewall_state), self._ATTACKER_TURN, owned_nodes)
            else:
                return (source_node, obtained_credentials, firewall_state, self._ATTACKER_TURN, owned_nodes)

    def final(self, state):
        return self._final_states

    def turn(self, state):
        return state[3]

    def enabled_acts(self, state):
        if state[3] == self._ATTACKER_TURN:
            return [action for action in self.actions() if self.is_attacker_action(action)]
        else:
            return [action for action in self.actions() if not self.is_attacker_action(action)]

    def is_attacker_action(self, action):
        if action == "no_attacker_action" or action[0:13] == "move_to_node_" or action[0:16] == "local_attack_on_" or action[0:11] == "connect_to_":
            return True
        else:
            return False
    def _construct_states(self):
        states = []  # list of state objects that can be used to construct game graph
        final_states = []

        unique_credentials = set()
        connections = []
        network_nodes = []
        for name, node in self._cbs_network.nodes.items():
            network_nodes.append(name)
            connections.extend([(name, target_node) for target_node in node["connected_nodes"]])
            unique_credentials.update(
                node["allowed_creds"])  # add credentials that can be used to connect to the node
            for leaked_credential in node["creds_stored"]:
                unique_credentials.update(leaked_credential[2])  # add credentials that are stored on the node

        possible_firewall_states = list(itertools.product([0, 1], repeat=len(connections)))
        possible_obtained_credentials_states = list(itertools.product([0, 1], repeat=len(unique_credentials)))
        possible_owned_nodes_configuration = list(itertools.product([0, 1], repeat=len(network_nodes)))

        for node_name in network_nodes:
            for firewall_state in possible_firewall_states:
                for obtained_credentials in possible_obtained_credentials_states:
                    # State is of form (id, i, bool[k], bool[n], turn, owned_nodes)
                    # Where i is the agent's location, k is the number of credentials, and n is the
                    # number of firewalls (connections between computers in the network, edges in the network graph)
                    for owned_nodes in possible_owned_nodes_configuration:
                        states.append((node_name, obtained_credentials, firewall_state, 1, owned_nodes))
                        states.append((node_name, obtained_credentials, firewall_state, 2, owned_nodes))
                        if node_name == self._final_node:
                            final_states.append((node_name, obtained_credentials, firewall_state, 1, owned_nodes))
                            final_states.append((node_name, obtained_credentials, firewall_state, 2, owned_nodes))
        return states, unique_credentials, possible_firewall_states, connections, final_states

    def _construct_actions(self):
        actions = []
        ### attacker actions ###
        # no action
        actions.append("no_attacker_action")
        for node in self._cbs_network.nodes:
            # change i to another owned node
            actions.append(f"move_to_node_{node}")
            # perform a local attack (add credentials from the current node i to the obtained credentials)
            actions.append(f"local_attack_on_{node}")
            # connect to a new node using obtained credentials
            for credential in self._credential_set:
                actions.append(f"connect_to_{node}_with_{credential}")
        ### defender actions ###
        # change firewall
        for firewall_state in self._possible_firewall_states:
            actions.append(f"change_firewall_to_{firewall_state}")
        return actions

if __name__ == '__main__':
    game = CBSGame("network.json", final_node="5")
    print(f"{len(game.states())=}")
    print(f"{len(game.actions())=}")
    # print(game.actions())
    state = ('0', (0, 0, 0), (1, 1, 1, 1, 1), 2, (1, 0, 0, 0, 0, 0))
    # print(game.delta(state, "move_to_node_0"))
    graph = game.graphify(pointed=True)
    print(f"{graph.number_of_nodes()=}")
    print(f"{graph.number_of_edges()=}")
    print(game.enabled_acts(state))

    ## trap_subsets is a dict with each network node and the graph nodes that become traps if that network node is a trap ##
    trap_subsets = {}
    for node in graph.nodes():
        # source_node is the name of the "computer" in the network that this state occurs in
        source_node = graph["state"][node][0]
        if source_node not in trap_subsets:
            trap_subsets[source_node] = []
        trap_subsets[source_node].append(node)
    fake_subsets = trap_subsets

    final_states = trap_subsets["0"]
    solver = SWinReach(graph, final=final_states)
    solver.solve()

    # why is this state not in p1's winning region?
    # It's because the attacker can just stay at node 2,
    # TODO We might get more interesting results if the attacker has to move?
    state = ('2', (0, 0, 0), (1, 1, 1, 1, 1), 2, (1, 0, 0, 0, 0, 0))
    # state2 = ('2', (0, 0, 0), (0, 0, 0, 0, 0), 1, (1, 0, 0, 0, 0, 0))
    print(game.enabled_acts(state))
    print(state in solver.win_region(1))
    print(game.delta(state, "move_to_node_0"))
    print(game.delta(state, "move_to_node_0") in solver.win_region(1))
    # sanity check trap_subsets
    print(trap_subsets.keys())
    for key in trap_subsets.keys():
        print(f"num states with node {key}: {len(trap_subsets[key])}")
        # for state in trap_subsets[key]:
        #     assert state[0] == key, f"Error trap_subset[{key}] contains state not at node {key}"
    arena_traps, arena_fakes, covered_states = greedy_max(graph, trap_subsets=trap_subsets, fake_subsets=None, max_traps=2)
    print(arena_traps, arena_fakes)

    # for every state1
        # for each incoming edge state2
            # for each incoming edge state3
                # if everything is the same between state3 and state1 except firewall_state and attacker location
                    # for every different attacker location in the set of state3s
                        # add an edge from state1 to every defender state (state with turn=2) with the new attacker location