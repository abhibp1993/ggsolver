from ggsolver.dtptb import SWinReach
"""
Algorithms and fact checking functions.
"""


def solve_game(game):
    """
    Solves the given game by applying the Zielonka's algorithm.
    :return: (:class:`dtptb.SureWinReach` instance) Solution of game.
    """
    pass


def check_fact1(p1_game, p2_game):
    """
    Checks second bullet point in "we note the following facts".
    Use assertions.
    """
    pass


def check_lemma19(hypergame):
    pass

def greedy_max(graph, trap_subsets, fake_subsets, defender_winning_states, max_traps=float("inf"), max_fakes=float("inf")):
    # trap_subsets is a mapping of an arena point to a list of states that are sure winning for p1 if that arena point is a trap
    # fake_subsets is a mapping of an arena point to a list of states that are sure winning for p1 if that arena point is a fake

    states = set()
    arena_points = set()
    for arena_point, state_list in trap_subsets.items():
        arena_points.add(arena_point)
        for state in state_list:
            states.add(state)

    arena_traps = set() # set of arena points
    arena_fakes = set() # set of arena points

    covered_states = set() # set of states
    trap_states = set() # set of states
    fake_states = set() # set of states

    iter_count = 0
    # Allocate traps
    while len(states - covered_states) > 0 and len(arena_traps) < max_traps:
        iter_count += 1
        print(f"Iteration {iter_count}")

        nontraps = arena_points - arena_traps
        updated_winning_regions = list()
        print(f"\tNon-traps: {nontraps}")

        for arena_point in nontraps:
            # the list of final states if this arena point is made into a trap
            final_states = list(trap_states) + trap_subsets[arena_point] + defender_winning_states

            solver = SWinReach(graph, final=final_states)
            solver.solve()
            # TODO add different metrics to determine value of arena point as a trap
            # pair = {"arena_point": arena_point, "value_of_trap": solver.win_region(1)}
            pair = { "arena_point": arena_point, "winning_states": solver.win_region(1) }

            updated_winning_regions.append(pair)

        next_trap = max(updated_winning_regions, key=lambda x: len(x["winning_states"]))
        arena_traps.add(next_trap["arena_point"])
        trap_states.update(trap_subsets[next_trap["arena_point"]])
        covered_states.update(next_trap["winning_states"])

        print(f"\tSelected Trap: {next_trap['arena_point']}")
        print(f"\tNew total trap states: {len(trap_states)}")
        print(f"\tNew total winning states: {len(covered_states)}")

    # Allocate fakes
    while len(states - covered_states) > 0 and len(arena_fakes) < max_fakes:
        iter_count += 1
        print(f"Iteration {iter_count}")

        # Potential points for fakes are all points that are not fakes OR TRAPS
        non_allocated_points = arena_points - arena_traps - arena_fakes
        updated_winning_regions = list()
        print(f"\tnon_allocated_points: {non_allocated_points}")

        for arena_point in non_allocated_points:
            # the list of final states if this arena point is made into a fake (we must include states made winning by traps)
            final_states = list(fake_states) + list(trap_states) + fake_subsets[arena_point]

            solver = SWinReach(graph, final=final_states)
            solver.solve()
            pair = { "arena_point": arena_point, "winning_states": solver.win_region(1) }

            updated_winning_regions.append(pair)

        # TODO what to do if two traps/fakes give the same number of winning states? does it matter which we pick?
        next_fake = max(updated_winning_regions, key=lambda x: len(x["winning_states"]))
        arena_fakes.add(next_fake["arena_point"])
        fake_states.update(fake_subsets[next_fake["arena_point"]])
        covered_states.update(next_fake["winning_states"])

        print(f"\tSelected Fake: {next_fake['arena_point']}")
        print(f"\tNew total fake states: {len(fake_states)}")
        print(f"\tNew total winning states: {len(covered_states)}")

    return arena_traps, arena_fakes, covered_states, trap_states, fake_states
