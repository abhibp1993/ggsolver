"""
API may include more options such as which properties to include in DOT file etc.
"""
import ggsolver
import networkx as nx
import pygraphviz
import re
from loguru import logger


def from_dot(fpath, graph, backend="rabinizer"):
    """
    Loads the graph from DOT file into given graph object.

    :param fpath:
    :param graph:
    :return:

    .. note:: `graph.clear()` is called first.
    """
    if backend == "rabinizer":
        return from_dot_rabinizer(fpath, graph)


def from_dot_rabinizer(fpath, graph):
    # Use networkx loader to load
    nx_graph = nx.nx_pydot.read_dot(fpath)

    graph.clear()
    graph.create_np("state", default="")
    graph.create_np("acc_set", default=set())
    graph.create_ep("label", default="")
    graph.create_gp("init_state")
    graph.create_gp("final")
    graph.create_gp("acc_cond")

    # Get label of generated dot file
    acc_str = nx_graph.graph["graph"]["label"]

    # Get acceptance condition
    match = re.search("Rabin", acc_str)
    if match:
        graph["acc_cond"] = "Rabin"

    # Get acceptance sets
    condition = r'(Fin\(\d+\) & Inf\(\d+\))'
    matches = re.findall(condition, acc_str)
    rabin_pairs = set()
    for cond in matches:
        fin_cond = r'Fin\(\d+\)'
        inf_cond = r'Inf\(\d+\)'
        fin_match = re.search(fin_cond, cond)
        inf_match = re.search(inf_cond, cond)
        fin_acc = re.findall(r'\d+', fin_match.group(0))
        inf_acc = re.findall(r'\d+', inf_match.group(0))
        rabin_pairs.add((int(fin_acc[0]), int(inf_acc[0])))
    graph["final"] = rabin_pairs

    node_mapping = dict()
    for node, data in nx_graph.nodes(data=True):
        is_invisible = "invis" == data.get("style", "")
        if is_invisible:
            continue
        uid = graph.add_node()
        node, node_acc_set = data["label"].replace('"', "").split("\\n")
        node_mapping[node] = uid
        graph["state"][uid] = f"q{node}"
        graph["acc_set"][uid] = node_acc_set

    for u, v, k, data in nx_graph.edges(data=True, keys=True):
        is_u_invisible = "invis" == nx_graph.nodes[u].get("style", "")
        if is_u_invisible:
            graph["init_state"] = graph["state"][node_mapping[v]]
            continue

        uid = node_mapping[u]
        vid = node_mapping[v]
        key = graph.add_edge(uid, vid)
        graph["label"][uid, vid, key] = data["label"].replace('"', "")

    return graph


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
        assert np in graph.node_properties, f"Node property {np} not found in {graph=}."
    for ep in edge_props:
        assert ep in graph.edge_properties, f"Edge property {ep} not found in {graph=}."

    # Construct dot file
    dot_lines = list()
    dot_lines.append("digraph G {")
    for uid in graph.nodes():
        # Set node properties
        node_properties = dict()

        # Define shape of node based on turn
        if "turn" in graph.node_properties:
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

    # Write to file
    with open(fpath, "w") as dot_file:
        dot_file.writelines(dot_lines)


def aut_to_dot(fpath, graph, node_props=None, edge_props=None, **kwargs):
    """
    Generates a DOT file from automaton graph. Uses automaton specific formatting.

    :param edge_props:
    :param node_props:
    :param fpath:
    :param graph:
    :return:
    """
    node_props = set() if node_props is None else set(node_props)
    edge_props = {'formula'}

    assert 'final' in graph.node_properties, f"Automaton graph missing node property: `final` ."
    assert 'formula' in graph.edge_properties, f"Automaton graph missing edge property: `formula`."
    assert 'init_state' in graph.graph_properties, f"Automaton graph missing node property: `init_state`."
    for np in node_props - {'final'}:
        assert np in graph.node_properties, f"Node property {np} not found in {graph=}."
    for ep in edge_props - {'formula'}:
        logger.warning(f"Ignoring edge property {ep} while converting automaton to dot. "
                       f"Only formula is displayed on edges")

    # Identify unique accepting state classes
    acc_classes = set()
    for uid in graph.nodes():
        acc_classes.update(graph['final'][uid])
    n_acc_classes = len(acc_classes)

    # Construct dot file
    dot_lines = list()
    dot_lines.append("digraph G {")
    dot_lines.append(
        f"NI [style=invisible];\n"
    )
    dot_lines.append(
        f"NI -> N0;\n"
    )
    for uid in graph.nodes():
        # Set node properties
        node_properties = dict()

        # Mark accepting states with double peripheries
        if len(graph['final'][uid]) > 0:
            node_properties["peripheries"] = 2

        # Construct label of node based on user provided node properties. Default show UID.
        if len(node_props) == 0:
            node_properties["label"] = f"{uid}"
        elif len(node_props) == 1:
            node_properties["label"] = f"{graph[list(node_props)[0]][uid]}"
        else:
            node_properties["label"] = "(" + ", ".join([str(graph[np][uid]) for np in node_props]) + ")"

        if len(graph['final'][uid]) > 0 and n_acc_classes > 1:
            node_properties["label"] += f"\n{graph['final'][uid]}"

        # Generate line in dot file for the node.
        dot_lines.append(
            f"N{uid} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
        )

        # Process outgoing edges from uid.
        for _, vid, key in graph.out_edges(uid):
            # Set node properties
            edge_properties = dict()

            # Construct label of edge based on user provided edge properties. Default empty string.
            edge_properties['label'] = graph['formula'][uid, vid, key]

            # Generate line in dot file for the node.
            dot_lines.append(
                f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
            )

    # Add final line to dot-lines
    dot_lines.append("}")

    # Write to file
    with open(fpath, "w") as dot_file:
        dot_file.writelines(dot_lines)


def solution_to_dot(fpath, graph, node_props=None, edge_props=None, color_scheme=None, **kwargs):
    """
    Generates a DOT file from solution graph. Uses solution specific formatting.

    :param node_props:
    :param edge_props:
    :param fpath:
    :param graph:
    :return:
    """
    node_props = set() if node_props is None else set(node_props) - {'node_winner'}
    edge_props = set() if edge_props is None else set(edge_props) - {'edge_winner'}
    color_scheme = {ggsolver.TURN_P1: 'blue', ggsolver.TURN_P2: 'red', ggsolver.TURN_NATURE: 'black'} \
        if color_scheme is None else color_scheme

    assert 'node_winner' in graph.node_properties, f"Automaton graph missing node property: `node_winner` ."
    assert 'edge_winner' in graph.edge_properties, f"Automaton graph missing edge property: `edge_winner`."
    for np in node_props - {'node_winner'}:
        assert np in graph.node_properties, f"Node property {np} not found in {graph=}."
    for ep in edge_props - {'edge_winner'}:
        assert ep in graph.edge_properties, f"Edge property {ep} not found in {graph=}."

    # Construct dot file
    dot_lines = list()
    dot_lines.append("digraph G {")
    for uid in graph.nodes():
        # Set node properties
        node_properties = dict()

        # Define color of node
        winner = graph['node_winner'][uid]
        node_properties["color"] = color_scheme[winner] if winner in color_scheme else "black"

        # Define shape of node based on turn
        if "turn" in graph.node_properties:
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
            node_properties["label"] = f"{graph[list(node_props)[0]][uid]}"
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

            # Define color of edge
            winner = graph['edge_winner'][uid, vid, key]
            edge_properties["color"] = color_scheme[winner] if winner in color_scheme else "black"

            # Construct label of edge based on user provided edge properties. Default empty string.
            if len(edge_props) == 0:
                edge_properties["label"] = ""
            elif len(edge_props) == 1:
                edge_properties["label"] = f"{graph[list(edge_props)[0]][uid, vid, key]}"
            else:
                edge_properties["label"] = "(" + ", ".join([str(graph[ep][uid, vid, key]) for ep in edge_props]) + ")"

            # Generate line in dot file for the node.
            dot_lines.append(
                f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
            )

    # Add final line to dot-lines
    dot_lines.append("}")

    # Write to file
    with open(fpath, "w") as dot_file:
        dot_file.writelines(dot_lines)


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
