"""
API may include more options such as which properties to include in DOT file etc.
"""


def from_dot(fpath, graph):
    """
    Loads the graph from DOT file into given graph object.

    :param fpath:
    :param graph:
    :return:

    .. note:: `graph.clear()` is called first.
    """
    pass


def to_dot(fpath, graph, formatting="simple"):
    """
    Generates a DOT file from graph.

    :param fpath:
    :param graph:
    :param formatting: Supported styles {'simple', 'aut', 'solution'}
    :return:
    """
    if formatting == "simple":
        graph_to_dot(fpath, graph)
    elif formatting == "simple":
        aut_to_dot(fpath, graph)
    elif formatting == "simple":
        solution_to_dot(fpath, graph)


def graph_to_dot(fpath, graph):
    """
    Generates a DOT file from given graph object. No special formatting is used.

    :param fpath:
    :param graph:
    :return:
    """
    raise NotImplementedError("Under development, graph_to_dot")


def aut_to_dot(fpath, graph):
    """
    Generates a DOT file from graph. Uses automaton specific formatting.

    :param fpath:
    :param graph:
    :return:
    """
    raise NotImplementedError("Under development, aut_to_dot")


def solution_to_dot(fpath, graph):
    """
    Generates a DOT file from graph. Uses solution specific formatting.

    :param fpath:
    :param graph:
    :return:
    """
    raise NotImplementedError("Under development, solution_to_dot")
