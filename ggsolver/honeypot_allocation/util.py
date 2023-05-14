import ggsolver.graph as ggraph
import pygraphviz
import os


def write_dot_file(graph: ggraph.Graph, path, filename, **kwargs):
    fpath = os.path.join(path, f"{filename}.dot")
    with open(fpath, 'w') as file:
        contents = list()
        contents.append("digraph G {\n")

        for node in graph.nodes():
            node_properties = {
                "shape": 'circle' if graph['turn'][node] == 1 else 'box',
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
    g.layout('dot')
    path = os.path.join(path, f"{filename}.svg")
    g.draw(path=path, format='svg')

