import random
import time

from ggsolver.decoy_alloc.models import *


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


if __name__ == '__main__':
    game = RandomReachabilityGame(num_states=100000, num_actions=10, num_final=10)

    start = time.perf_counter()
    game.graphify(parallel=False)
    end = time.perf_counter()
    print(f"Unpointed single-core graphify in {end - start} seconds.")

    start = time.perf_counter()
    game.graphify(parallel=True)
    end = time.perf_counter()
    print(f"Unpointed multi-core graphify in {end - start} seconds.")

    start = time.perf_counter()
    game.initialize("s0")
    game.graphify(pointed=True, parallel=False)
    end = time.perf_counter()
    print(f"Pointed single-core graphify in {end - start} seconds.")

    start = time.perf_counter()
    game.initialize("s0")
    game.graphify(pointed=True, parallel=True)
    end = time.perf_counter()
    print(f"Pointed multi-core graphify in {end - start} seconds.")

