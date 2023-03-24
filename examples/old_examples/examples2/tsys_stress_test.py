import random
from ggsolver.models import TSys
from time import perf_counter

NUM_NODES = 10**5


class MyTSys(TSys):
    def states(self):
        return range(NUM_NODES)

    def actions(self):
        return ["a", "b"]

    def delta(self, state, inp):
        return random.randint(0, NUM_NODES - 1)


if __name__ == '__main__':
    tsys = MyTSys()
    tsys.initialize(1)

    t1_start = perf_counter()
    # tsys.graphify()
    tsys.graphify(pointed=True)
    t1_stop = perf_counter()

    print(f"time: {t1_stop - t1_start} sec")
