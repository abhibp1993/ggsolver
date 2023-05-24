import ggsolver.honeypot_allocation.game_generator as gen
import ggsolver.honeypot_allocation.solvers as solvers
import random
import sys
from ggsolver.dtptb.dtptb_reach import SWinReach
from loguru import logger

logger.remove()
logger.add("toy_problem_search.debug.log", level="DEBUG")
logger.add("toy_problem_search.success.log", level="SUCCESS")
logger.add(sys.stdout, level="ERROR")


def search(num_nodes, max_out_degree, num_final, rnd_start=0, rnd_end=100):
    for seed in range(rnd_start, rnd_end):
        try:
            # Set random seed
            random.seed(seed)

            # Define output folders
            dirname_t2_f0 = f"out/out_{seed}_t2_f0"
            dirname_t1_f1 = f"out/out_{seed}_t1_f1"
            dirname_t0_f2 = f"out/out_{seed}_t0_f2"

            # Construct game
            game = gen.Hybrid(num_nodes=num_nodes, max_out_degree=max_out_degree, num_final=num_final)
            game_graph = game.graphify()

            # TODO. Check whether P2 winning region is trivial. If yes, reject this example.
            game_solution = SWinReach(game_graph, player=2)
            game_solution.solve()
            if len(game_solution.winning_nodes(2)) < 0.5 * game_graph.number_of_nodes():
                logger.critical(f"Rejecting seed: {seed}. {len(game_solution.winning_nodes(2))} / {game_graph.number_of_nodes()}")
                continue

            # Allocate decoys
            alloc_t2_f0 = solvers.DecoyAllocator(game_graph, num_traps=2, num_fakes=0, debug=True, path=dirname_t2_f0)
            alloc_t2_f0.solve()
            alloc_t2_f0.save_dot(dirname_t2_f0, filename="colored_graph")
            alloc_t2_f0.save_svg(dirname_t2_f0, filename="colored_graph")

            alloc_t1_f1 = solvers.DecoyAllocator(game_graph, num_traps=1, num_fakes=1, debug=True, path=dirname_t1_f1)
            alloc_t1_f1.solve()
            alloc_t1_f1.save_dot(dirname_t1_f1, filename="colored_graph")
            alloc_t1_f1.save_svg(dirname_t1_f1, filename="colored_graph")

            alloc_t0_f2 = solvers.DecoyAllocator(game_graph, num_traps=0, num_fakes=2, debug=True, path=dirname_t0_f2)
            alloc_t0_f2.solve()
            alloc_t0_f2.save_dot(dirname_t0_f2, filename="colored_graph")
            alloc_t0_f2.save_svg(dirname_t0_f2, filename="colored_graph")

            print(f"RND: {seed}, VOD(T2, F0): {alloc_t2_f0.vod()}, VOD(T1, F1): {alloc_t1_f1.vod()}, VOD(T0, F2): {alloc_t0_f2.vod()}")

        except:
            pass


if __name__ == '__main__':
    with logger.catch():
        search(num_nodes=12, max_out_degree=3, num_final=2, rnd_start=63, rnd_end=84)
