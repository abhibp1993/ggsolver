"""
The example used here is from Principles of Model Checking, Fig. 10.21.
This is not suitable for studying preferences. But is used as a quick test for code.
"""

from ggsolver.pbp.safeimp.models import *
from ggsolver.pbp.safeimp.reachability import *


if __name__ == '__main__':
    mdp = QualitativeMDP(
        states=[f"s{i}" for i in range(8)] + ["sink"],
        actions=['alpha', 'beta'],
        trans_dict={
            "s0": {'alpha': ["s1"], 'beta': ["s2", "s4"]},
            "s1": {'alpha': ["s1", "s2", "s3"], 'beta': ["sink"]},
            "s2": {'alpha': ["s2"], 'beta': ["sink"]},
            "s3": {'alpha': ["s3"], 'beta': ["sink"]},
            "s4": {'alpha': ["s5", "s6"], 'beta': ["sink"]},
            "s5": {'alpha': ["s6"], 'beta': ["s2", "s7"]},
            "s6": {'alpha': ["s5", "s6"], 'beta': ["sink"]},
            "s7": {'alpha': ["s2", "s3"], 'beta': ["sink"]},
            "sink": {'alpha': ["sink"], 'beta': ["sink"]},
        },
        init_state="s0",
        final=["s6"]
        # final=["s6", "s7"]
        # final=["s2", "s3"]
    )

    pref = PrefModel(
        outcomes={1: ["s6", "s7"], 2: ["s7"]},
        pref=[(1, 2)]
    )

    imdp = ImprovementMDP(mdp, pref)
    imdp_graph = imdp.graphify()
    imdp_graph.to_png("imdp_graph.png", nlabel=["state"], elabel=["input"])
    final_nodes = {node for node in imdp_graph.nodes() if imdp_graph["state"][node][1] == 1}
    sasi = SASIReach(imdp_graph, final=final_nodes)
    sasi.solve()