"""
API may include more options such as which properties to include in DOT file etc.
"""
import ggsolver
import pygraphviz


def from_dot(fpath, graph):
    """
    Loads the graph from DOT file into given graph object.

    :param fpath:
    :param graph:
    :return:

    .. note:: `graph.clear()` is called first.
    """
    pass


def to_dot(fpath, graph, formatting="simple", node_props=None, edge_props=None, **kwargs):
    """
    Generates a DOT file from graph.

    :param edge_props:
    :param node_props:
    :param fpath:
    :param graph:
    :param formatting: Supported styles {'simple', 'aut', 'solution'}
    :return:
    """
    if formatting == "simple":
        graph_to_dot(fpath, graph, node_props=node_props, edge_props=edge_props, **kwargs)
    elif formatting == "aut":
        aut_to_dot(fpath, graph, node_props=node_props, edge_props=edge_props, **kwargs)
    elif formatting == "solution":
        solution_to_dot(fpath, graph, node_props=node_props, edge_props=edge_props, **kwargs)


def graph_to_dot(fpath, graph, node_props=None, edge_props=None, **kwargs):
    """
    Generates a DOT file from given graph object. No special formatting is used.

    :param edge_props:
    :param node_props:
    :param fpath:
    :param graph:
    :return:
    """
    node_props = list() if node_props is None else node_props
    edge_props = list() if edge_props is None else edge_props
    for np in node_props:
        assert np in graph.np, f"Node property {np} not found in {graph=}."
    for ep in edge_props:
        assert ep in graph.ep, f"Edge property {ep} not found in {graph=}."

    # Construct dot file
    dot_lines = list()
    dot_lines.append("digraph G {")
    for uid in graph.nodes():
        # Set node properties
        node_properties = dict()

        # Define shape of node based on turn
        if "turn" in graph.np:
            if graph['turn'][uid] == ggsolver.TURN_P1:
                node_properties["shape"] = 'circle'
            elif graph['turn'][uid] == ggsolver.TURN_P2:
                node_properties["shape"] = 'box'
            elif graph['turn'][uid] == ggsolver.TURN_NATURE:
                node_properties["shape"] = 'diamond'
            else:
                node_properties["shape"] = 'ellipse'

        # Construct label of node based on user provided node properties. Default show UID.
        if len(node_props) == 0:
            node_properties["label"] = f"{uid}"
        elif len(node_props) == 1:
            node_properties["label"] = f"{graph[node_props[0]][uid]}"
        else:
            node_properties["label"] = "(" + ", ".join([str(graph[np][uid]) for np in node_props]) + ")"

        # Generate line in dot file for the node.
        dot_lines.append(
            f"N{uid} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
        )

        # Process outgoing edges from uid.
        for _, vid, key in graph.out_edges(uid):
            # Set node properties
            edge_properties = dict()

            # Construct label of edge based on user provided edge properties. Default empty string.
            if len(edge_props) == 0:
                edge_properties["label"] = ""
            elif len(edge_props) == 1:
                edge_properties["label"] = f"{graph[edge_props[0]][uid, vid, key]}"
            else:
                edge_properties["label"] = "(" + ", ".join([str(graph[ep][uid, vid, key]) for ep in edge_props]) + ")"

            # Generate line in dot file for the node.
            dot_lines.append(
                f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
            )

    # Add final line to dot-lines
    dot_lines.append("}")

    # # Append new line character to all entries
    # dot_lines = [line + "\n" for line in dot_lines]

    # Write to file
    with open(fpath, "w") as dot_file:
        dot_file.writelines(dot_lines)


def aut_to_dot(fpath, graph, node_props=None, edge_props=None, **kwargs):
    """
    Generates a DOT file from graph. Uses automaton specific formatting.

    :param edge_props:
    :param node_props:
    :param fpath:
    :param graph:
    :return:
    """
    raise NotImplementedError("Under development, aut_to_dot")


def solution_to_dot(fpath, graph, node_props=None, edge_props=None, **kwargs):
    """
    Generates a DOT file from graph. Uses solution specific formatting.

    :param node_props:
    :param edge_props:
    :param fpath:
    :param graph:
    :return:
    """
    raise NotImplementedError("Under development, solution_to_dot")


def dot2svg(dot_fpath, svg_fpath, layout_engine="dot", node_props=None, edge_props=None, **kwargs):
    """
    Converts dot file to SVG file.
    :param layout_engine: Only support "dot" for not.
    :param dot_fpath: Complete file path including extension of DOT file.
    :param svg_fpath: Complete file path including extension of SVG file.
    """
    g = pygraphviz.AGraph(dot_fpath)
    g.layout('dot')
    g.draw(path=svg_fpath, format='svg')


def dot2png(dot_fpath, png_fpath, layout_engine="dot"):
    """
    Converts dot file to SVG file.
    :param layout_engine: Only support "dot" for not.
    :param dot_fpath: Complete file path including extension of DOT file.
    :param png_fpath: Complete file path including extension of SVG file.
    """
    g = pygraphviz.AGraph(dot_fpath)
    g.layout('dot')
    g.draw(path=png_fpath, format='png')


def dot2pdf(dot_fpath, pdf_fpath, layout_engine="dot"):
    """
    Converts dot file to SVG file.
    :param layout_engine: Only support "dot" for not.
    :param dot_fpath: Complete file path including extension of DOT file.
    :param pdf_fpath: Complete file path including extension of SVG file.
    """
    g = pygraphviz.AGraph(dot_fpath)
    g.layout('dot')
    g.draw(path=pdf_fpath, format='pdf')
