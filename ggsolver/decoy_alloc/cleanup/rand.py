import random

from ggsolver.decoy_alloc.cleanup.models import *


class RandomReachabilityGame(ReachabilityGame):
    def __init__(self, num_states, num_actions, num_final):
        super(RandomReachabilityGame, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_final = num_final

    def states(self):
        return [f"s{i}" for i in range(self.num_states)]

    def actions(self):
        return [f"a{i}" for i in range(self.num_actions)]

    def delta(self, state, act):
        return f"s{random.randint(0, self.num_states - 1)}"

    def final(self, state):
        return random.sample(range(self.num_states), k=self.num_final)

    def turn(self, state):
        return random.randint(0, 1)

