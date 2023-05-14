import ggsolver.graph as ggraph
import pygraphviz
import os
from loguru import logger
from tqdm import tqdm


def write_dot_file(graph: ggraph.Graph, path, filename, **kwargs):
    fpath = os.path.join(path, f"{filename}.dot")
    with open(fpath, 'w') as file:
        contents = list()
        contents.append("digraph G {\noverlap=scale;\n")

        for node in tqdm(graph.nodes(), desc=f"Generating DOT file: {fpath}."):
            node_properties = {
                "shape": 'circle' if graph['turn'][node] == 1 else 'box',
                "style": 'filled',
                "width": 2,
                "height": 2,
                "label": graph['state'][node],
                "peripheries": '2' if graph['final'][node] else '1',
            }
            if "node_winner" in graph.node_properties:
                node_properties |= {"color": 'blue' if graph['node_winner'][node] == 1 else 'red'}

            contents.append(
                f"N{node} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
            )

            for uid, vid, key in graph.out_edges(node):
                edge_properties = {
                    "label": graph["input"][uid, vid, key] if kwargs.get("no_actions", False) else ""
                }
                if "edge_winner" in graph.edge_properties:
                    if graph['edge_winner'][uid, vid, key] == 1:
                        edge_properties |= {"color": 'blue'}
                    elif graph['edge_winner'][uid, vid, key] == 2:
                        edge_properties |= {"color": 'red'}
                    else:
                        edge_properties |= {"color": 'black'}

                contents.append(
                    f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
                )

        contents.append("}")
        file.writelines(contents)

    # Generate SVG
    g = pygraphviz.AGraph(fpath)
    if graph.number_of_nodes() + graph.number_of_edges() > 200:
        logger.warning(f"Graph size is larger than 200, using Force-Directed Layout. Generating PNG instead of SVG.")
        g.layout('sfdp')
        path = os.path.join(path, f"{filename}.png")
        g.draw(path=path, format='png')
    else:
        g.layout('dot')
        path = os.path.join(path, f"{filename}.svg")
        g.draw(path=path, format='svg')

    # g.layout('dot')
    # path = os.path.join(path, f"{filename}.svg")
    # g.draw(path=path, format='svg')

