import json
import ggsolver.graph as ggraph
from networkx.drawing.nx_agraph import read_dot
from loguru import logger

FILE_GGRAPH = "out/ex14_5x5_UAV_UGV/0_1/ex14_5x5_UAV_UGV_0_1_p1.ggraph"
FILE_DOT = "out/ex14_5x5_UAV_UGV/0_1/ex14_5x5_UAV_UGV_0_1_p1.dot"
FILE_VB = "out/ex14_5x5_UAV_UGV/0_1/ex14_5x5_UAV_UGV_0_1_p1.json"


def gen_vb_output():
    # Initialize output dictionary for VB
    out = {"state": dict(), "transitions": dict(), "win": list(), "init_state": None}

    # Load dot file
    logger.info("Loading DOT file...")
    dot_graph = read_dot(FILE_DOT)

    # Load JSON ggraph file
    logger.info("Loading ggsolver graph...")
    graph = ggraph.Graph().load(FILE_GGRAPH)

    # Iterate over nodes to extract information
    state2node = dict()
    for node, data in dot_graph.nodes(data=True):
        # Get node id
        uid = int(node[1:])
        state2node[graph["state"][uid]] = uid 

        # Store the state information in VB output
        out["state"][uid] = graph["state"][uid]

        # If node is winning for P1, mark it.
        if data["color"] == "green":
            out["win"].append(uid)

        # Mark edge winners
        for _, vid, key, data in dot_graph.out_edges(f"N{uid}", keys=True, data=True):
            vid = int(vid[1:])
            if data["color"] == "green":
                act = graph["input"][uid, vid, key]
                if uid not in out["transitions"]:
                    out["transitions"][uid] = {act: vid}
                else:
                    out["transitions"][uid].update({act: vid})

        # Mark initial state
        out["init_state"] = state2node[graph["init_state"]]

    with open(FILE_VB, "w") as file:
        json.dump(out, file, indent=2)


if __name__ == '__main__':
    gen_vb_output()
