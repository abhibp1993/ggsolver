import ggsolver.dtptb as dtptb
import ggsolver.honeypot_allocation.util as util
from ggsolver.dtptb.dtptb_reach import SWinReach
from ggsolver.honeypot_allocation.solvers import *
from loguru import logger


class ToyProblem(dtptb.DTPTBGame):
    def __init__(self):
        super(ToyProblem, self).__init__()
        self._trans_dict = {
            "s0": {"b1": "s0", "b2": "s0"},
            "s1": {"a1": "s0", "a2": "s0"},
            "s2": {"b1": "s5", "b2": "s0"},
            "s3": {"a1": "s0", "a2": "s1"},
            "s4": {"a1": "s1", "a2": "s1"},
            "s5": {"b1": "s7", "b2": "s2"},
            "s6": {"b1": "s3", "b2": "s4"},
            "s7": {"a1": "s5", "a2": "s6"},
            "s8": {"b1": "s7", "b2": "s8"},
            "s9": {"a1": "s8", "a2": "s6"},
            "s10": {"a1": "s8", "a2": "s10"},
            "s11": {"b1": "s10", "b2": "s10"},
        }

    def __str__(self):
        delta = {(st, a): self.delta(st, a) for st in self.states() for a in self.enabled_acts(st)}
        final = {st for st in self.states() if self.final(st)}
        return f"{self.states()=}\n" \
               f"{delta=}\n" \
               f"{final=}\n"

    def states(self):
        return [f"s{i}" for i in range(12)]

    def turn(self, state):
        turn_map = {
            "s0": 2,
            "s1": 1,
            "s2": 2,
            "s3": 1,
            "s4": 1,
            "s5": 2,
            "s6": 2,
            "s7": 1,
            "s8": 2,
            "s9": 1,
            "s10": 1,
            "s11": 2,
        }
        return turn_map[state]

    def actions(self):
        return ["a1", "a2", "b1", "b2"]

    def enabled_acts(self, state):
        return list(self._trans_dict[state].keys())

    def delta(self, state, act):
        return self._trans_dict.get(state, dict()).get(act, None)

    def final(self, state):
        if state in ["s0", "s1"]:
            return True
        return False


if __name__ == '__main__':
    # Create instance of ToyProblem and solve it.
    game = ToyProblem()
    game_graph = game.graphify()
    swin = SWinReach(game_graph, player=2)
    swin.solve()
    util.write_dot_file(swin.solution(), "out/", "toy_problem", show_actions=True)

    # Construct state2node map
    state2node = {game_graph["state"][uid]: uid for uid in game_graph.nodes()}

    # Place trap at s7 and solve it.
    fdir = "out/trap_s7/"
    dswin = DSWinReach(game_graph, traps={state2node["s7"]}, fakes=set(), debug=True, path=fdir)
    dswin.solve()
    dswin.save_svg(fdir, filename="trap_s7_dswin", show_actions=True)
    logger.info(f"VOD: {dswin.vod()}")

    # Place fake at s7 and solve it.
    fdir = "out/fake_s7/"
    dswin = DSWinReach(game_graph, traps=set(), fakes={state2node["s7"]}, debug=True, path=fdir)
    dswin.solve()
    dswin.save_svg(fdir, filename="fake_s7_dswin", show_actions=True)
    logger.info(f"VOD: {dswin.vod()}")

    # Best allocator for 2 traps.
    fdir = "out/trap_allocator/"
    alloc = DecoyAllocator(game_graph, num_traps=2, num_fakes=0, debug=True, path=fdir, filename="trap_allocator")
    alloc.solve()

    # Best allocator for 2 fakes.
    fdir = "out/fake_allocator/"
    alloc = DecoyAllocator(game_graph, num_traps=0, num_fakes=2, debug=True, path=fdir, filename="fake_allocator")
    alloc.solve()

    # Best allocator for 1 trap, 1 fake.
    fdir = "out/fake_trap_allocator/"
    alloc = DecoyAllocator(game_graph, num_traps=1, num_fakes=1, debug=True, path=fdir, filename="fake_trap_allocator")
    alloc.solve()
