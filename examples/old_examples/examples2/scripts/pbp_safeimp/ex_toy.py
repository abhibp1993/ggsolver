"""
The example used here is from Principles of Model Checking, Fig. 10.21.
This is not suitable for studying preferences. But is used as a quick test for code.
"""

from ggsolver.pbp.safeimp.models import *
from ggsolver.pbp.safeimp.reachability import *
import logging
logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    mdp = QualitativeMDP(
        states=[f"s{i}" for i in range(8)],
        actions=['a', 'b', 'c'],
        trans_dict={
            "s0": {'a': ["s2"], 'b': ["s3", "s4"], 'c': ["s1"]},
            "s1": {'a': ["s1"], 'b': ["s1"], 'c': ["s1"]},
            "s2": {'a': ["s2"], 'b': ["s2"], 'c': ["s2"]},
            "s3": {'a': ["s5", "s6"], 'b': ["s7"], 'c': ["s3"]},
            "s4": {'a': ["s4"], 'b': ["s5", "s6"], 'c': ["s4"]},
            "s5": {'a': ["s5"], 'b': ["s5"], 'c': ["s5"]},
            "s6": {'a': ["s6"], 'b': ["s6"], 'c': ["s6"]},
            "s7": {'a': ["s7"], 'b': ["s7"], 'c': ["s7"]},
        },
        init_state="s0",
        final=["s6"]
        # final=["s6", "s7"]
        # final=["s2", "s3"]
    )

    pref = PrefModel(
        outcomes={1: ["s2"], 2: ["s3"], 3: ["s4"], 4: ["s5"], 5: ["s6"]},
        pref=[(2, 1), (3, 1), (4, 1), (5, 1), (4, 2), (5, 2), (4, 3), (5, 3)]
    )

    imdp = ImprovementMDP(mdp, pref)
    imdp_graph = imdp.graphify()
    imdp_graph.to_png("imdp_graph.png", nlabel=["state"], elabel=["input"])
    final_nodes = {node for node in imdp_graph.nodes() if imdp_graph["state"][node][1] == 1}
    sasi = SASIReach(imdp_graph, final=final_nodes)
    sasi.solve()
