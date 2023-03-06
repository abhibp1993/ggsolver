import ggsolver.dtptb as dtptb
from ggsolver.logic.ltl import ScLTL


states = list(range(3))
actions = ["a", "b"]
trans_dict = {
    0: {"a": 1, "b": 2},
    1: {"a": 1, "b": 1},
    2: {"a": 2, "b": 0},
}
atoms = [f"p{i}" for i in states]
label = {i: [f"p{i}"] for i in states}
turn = {
    0: 2,
    1: 2,
    2: 1,
}

game = dtptb.DTPTBGame(states=states, actions=actions, trans_dict=trans_dict, atoms=atoms, label=label, turn=turn)
objective = ScLTL("Fp1", atoms=atoms)
dfa = objective.translate()
for state in dfa.states():
    print(f"{state=}, {dfa.final(state)=}")
print()

prod_game = dtptb.ProductWithDFA(game, dfa)
for state in prod_game.states():
    print(f"{state=}, {prod_game.final(state)=}")

graph = prod_game.graphify()
swin = dtptb.SWinReach(graph)
swin.solve()
for node in graph.nodes():
    state = graph['state'][node]
    print(f"{node=}, {state=}{swin.winner(state)=}")

for node in graph.nodes():
    state = graph['state'][node]
    print(f"{node=}, {state=}, {swin.win_acts(state)=}")
