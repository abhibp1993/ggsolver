import itertools

import networkx as nx


def stochastic_weak_order(preference_graph: nx.MultiDiGraph):
    ordering = dict()
    for v in preference_graph.nodes:
        # Compute upper closure of {v}
        upper_closure_v = set(nx.descendants(preference_graph, v)) | {v}

        # Extract set of semi-automaton states belonging to a preference graph node in the upper-closure
        ordering[v] = set.union(*[set(preference_graph.nodes[node]["partition"]) for node in upper_closure_v])
    # Return ordering
    return ordering


def stochastic_weak_star_order(preference_graph: nx.MultiDiGraph):
    ordering = dict()
    for v in preference_graph.nodes:
        # Compute E \ lower_closure({v})
        lower_closure_v = set(nx.ancestors(preference_graph, v)) | {v}
        non_lower_closure_v = set(preference_graph.nodes()) - lower_closure_v

        # Extract set of semi-automaton states belonging to a preference graph node in E \ lower_closure({v})
        if len(non_lower_closure_v) == 0:
            ordering[v] = set()
        else:
            ordering[v] = set.union(
                *[set(preference_graph.nodes[node]["partition"]) for node in non_lower_closure_v])

    # Return ordering
    return ordering


def stochastic_strong_order(preference_graph: nx.MultiDiGraph):
    if preference_graph.number_of_nodes() > 8:
        raise ValueError("Computing strong stochastic order is not supported for preference graph "
                         "with 8+ nodes due to subset construction.")

    ordering = dict()
    for n in range(preference_graph.number_of_nodes() + 1):
        for subset in itertools.combinations(preference_graph.nodes, n):
            # Compute upper closure of `subset` of nodes (i.e., nodes reachable from at least one state in subset)
            upper_closure_subset = set()
            for layer in nx.bfs_layers(preference_graph, sources=subset):
                upper_closure_subset.update(layer)

            # Extract set of semi-automaton states belonging to some preference graph node in upper-closure
            if len(upper_closure_subset) == 0:
                ordering[subset] = set()
            else:
                ordering[subset] = set.union(
                    *[set(preference_graph.nodes[node]["partition"]) for node in upper_closure_subset]
                )

    # Return ordering
    return ordering


def build_stochastic_order_objectives(model, ordering_func):
    ordering = ordering_func(model.aut.pref_graph)

    ordering_vector = [
        set(k) for k, v in
        sorted({tuple(sorted(v)): k for k, v in ordering.items()}.items(), key=lambda x: x[1])
    ]

    objective = [set() for _ in range(len(ordering_vector))]
    for state in model.states():
        for i, aut_obj_i in enumerate(ordering_vector):
            if state.aut_state in aut_obj_i:
                objective[i].add(state)

    return objective, ordering_vector
