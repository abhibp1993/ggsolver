"""
ggsolver: graph.py
License goes here...
"""

import json
import os
import pickle
from functools import reduce

import networkx as nx
from ggsolver import util


class IGraph:
    """
    Graph interface. A graphical model must implement a IGraph interface.
    """
    def __init__(self):
        self._graph = None
        self._node_properties = dict()
        self._edge_properties = dict()
        self._graph_properties = dict()

    def __getitem__(self, pname):
        if pname in self._node_properties:
            return self._node_properties[pname]
        elif pname in self._edge_properties:
            return self._edge_properties[pname]
        elif pname in self._graph_properties:
            return self._graph_properties[pname]
        else:
            raise KeyError(f"{pname} is not a valid node/edge/graph property.")

    def __setitem__(self, pname, pmap):
        # Check if the property exists
        if pname in self._node_properties.keys():
            assert isinstance(pmap, NodePropertyMap)

        if pname in self._edge_properties.keys():
            assert isinstance(pmap, EdgePropertyMap)

        # Update property value
        if isinstance(pmap, NodePropertyMap):
            pmap.graph = self
            self._node_properties[pname] = pmap
        elif isinstance(pmap, EdgePropertyMap):
            pmap.graph = self
            self._edge_properties[pname] = pmap
        else:
            self._graph_properties[pname] = pmap

    @property
    def node_properties(self):
        """ Returns the node properties as a dictionary of {"property name": NodePropertyMap object}. """
        return self._node_properties

    @property
    def edge_properties(self):
        """ Returns the edge properties as a dictionary of {"property name": EdgePropertyMap object}. """
        return self._edge_properties

    @property
    def graph_properties(self):
        """ Returns the graph properties as a dictionary of {"property name": property value}. """
        return self._graph_properties

    def add_node(self):
        pass

    def add_nodes(self, num_nodes):
        pass

    def add_edge(self, uid, vid):
        pass

    def add_edges(self, edges):
        """ (uid, vid) pairs """
        pass

    def rem_node(self, uid):
        pass

    def rem_edge(self, uid, vid, key):
        pass

    def has_node(self, uid):
        pass

    def has_edge(self, uid, vid, key=None):
        pass

    def nodes(self):
        pass

    def edges(self):
        pass

    def successors(self, uid):
        pass

    def predecessors(self, uid):
        pass

    def neighbors(self, uid):
        pass

    def ancestors(self, uid):
        pass

    def descendants(self, uid):
        pass

    def in_edges(self, uid):
        pass

    def out_edges(self, uid):
        pass

    def number_of_nodes(self):
        pass

    def number_of_edges(self):
        pass

    def clear(self):
        pass

    def serialize(self):
        pass

    def deserialize(self, obj_dict):
        pass

    def has_property(self, pname):
        if pname in self._node_properties or pname in self._edge_properties or pname in self._graph_properties:
            return True
        return False

    def save(self, fpath, overwrite=False, protocol="json"):
        pass

    def load(self, fpath, protocol="json"):
        pass


class NodePropertyMap(dict):
    """
    Implements a default dictionary that maps a node ID to its property value. To store data efficiently,
    only the non-default values are stored in the dictionary.

    Raises an error if the node ID is invalid.
    """
    def __init__(self, graph, default=None):
        super(NodePropertyMap, self).__init__()
        self.graph = graph
        self.default = default

    def __repr__(self):
        return f"<NodePropertyMap graph={repr(self.graph)}>"

    def __missing__(self, node):
        if self.graph.has_node(node):
            return self.default
        raise ValueError(f"[ERROR] NodePropertyMap.__missing__:: {repr(self.graph)} does not contain node {node}.")

    def __getitem__(self, node):
        try:
            return super(NodePropertyMap, self).__getitem__(node)
        except KeyError:
            return self.__missing__(node)

    def __setitem__(self, node, value):
        assert self.graph.has_node(node), f"Node {node} not in {self.graph}."
        if value != self.default:
            super(NodePropertyMap, self).__setitem__(node, value)

    def serialize(self):
        return {
            "default": self.default,
            "dict": {k: v for k, v in self.items()}
        }

    def deserialize(self, obj_dict):
        self.clear()
        self.default = obj_dict["default"]
        # Explicitly deserialize to ensure all keys are valid nodes.
        for k, v in obj_dict["dict"].items():
            self[int(k)] = v


class EdgePropertyMap(dict):
    """
    Implements a default dictionary that maps an edge (uid, vid, key) to its property value. To store data efficiently,
    only the non-default values are stored in the dictionary.

    Raises an error if the edge (uid, vid, key) is invalid.
    """
    def __init__(self, graph, default=None):
        super(EdgePropertyMap, self).__init__()
        self.graph = graph
        self.default = default

    def __repr__(self):
        return f"<EdgePropertyMap graph={repr(self.graph)}>"

    def __missing__(self, edge):
        if self.graph.has_edge(*edge):
            return self.default
        raise ValueError(f"[ERROR] EdgePropertyMap.__missing__:: {repr(self.graph)} does not contain node {edge}.")

    def __getitem__(self, edge):
        try:
            return dict.__getitem__(self, edge)
        except KeyError:
            return self.__missing__(edge)

    def __setitem__(self, node, value):
        if value != self.default:
            super(EdgePropertyMap, self).__setitem__(node, value)

    def serialize(self):
        return {
            "default": self.default,
            "dict": [{"edge": edge, "pvalue": pvalue} for edge, pvalue in self.items()]
        }

    def deserialize(self, obj_dict):
        self.clear()
        self.default = obj_dict["default"]
        # Explicitly deserialize to ensure all keys are valid edges.
        for item in obj_dict["dict"]:
            self[tuple(item["edge"])] = item["pvalue"]


class Graph(IGraph):
    """
    A MultiDiGraph class represented as a 5-tuple (nodes, edges, node_properties, edge_properties, graph_properties).
    In addition, the graph implements a serialization protocol, save-load and drawing functionality.
    """
    def __init__(self):
        super(Graph, self).__init__()
        self._graph = nx.MultiDiGraph()

    def __str__(self):
        return f"<Graph with |V|={self.number_of_nodes()}, |E|={self.number_of_edges()}>"

    def base_graph(self):
        return self._graph

    def add_node(self):
        """
        Adds a new node to the graph.

        :warning: Duplication is NOT checked.

        :return: (int) ID of the added node.
        """
        uid = self._graph.number_of_nodes()
        self._graph.add_node(uid)
        return uid

    def add_nodes(self, num_nodes):
        """
        Adds multiple nodes to the graph.

        :warning: Duplication is NOT checked.
        :param num_nodes: (int) Number of nodes to be added.
        :return: (list) IDs of added nodes.
        """
        return [self.add_node() for _ in range(num_nodes)]

    def add_edge(self, uid, vid):
        """
        Adds a new edge between the give nodes.

        :warning: Duplication is NOT checked. Hence, calling the function twice adds two parallel edges between
            the same nodes.

        :return: (int) Key of the added edge. Key = 0 means the first edge was added between the given nodes.
            If Key = k, then (k+1)-th edge was added.
        """
        return self._graph.add_edge(uid, vid)

    def add_edges(self, edges):
        """
        Adds multiple edge between the give nodes.

        :warning: Duplication is NOT checked. Hence, calling the function twice adds two parallel edges between
            the same nodes.

        :return: (int) Key of the added edge. Key = 0 means the first edge was added between the given nodes.
            If Key = k, then (k+1)-th edge was added.
        """
        return [self.add_edge(uid, vid) for uid, vid in edges]

    def rem_node(self, uid):
        """
        Removal of nodes is NOT supported. Use filtering instead.
        """
        raise NotImplementedError("Removal of nodes is not supported. Use SubGraph instead.")

    def rem_edge(self, uid, vid, key):
        """
        Removal of edges is NOT supported. Use filtering instead.
        """
        raise NotImplementedError("Removal of nodes is not supported. Use SubGraph instead.")

    def has_node(self, uid):
        """
        Checks whether the graph has the given node or not.

        :param uid: (int) Node ID to be checked for containment.
        :return: (bool) True if given node is in the graph, else False.
        """
        return self._graph.has_node(uid)

    def has_edge(self, uid, vid, key=None):
        """
        Checks whether the graph has the given edge or not.

        :param uid: (int) Source node ID.
        :param vid: (int) Target node ID.
        :param key: If provided, checks whether the edge (u, v, k) is in the graph or not. Otherwise, checks if there
            exists an edge between nodes represented by uid and vid.
        :type key: int, optional
        :return: (bool) True if given edge is in the graph, else False.
        """
        return self._graph.has_edge(uid, vid, key)

    def nodes(self):
        """
        List of all nodes in the graph.
        """
        return list(self._graph.nodes())

    def edges(self):
        """
        List of all edges in the graph. Each edge is represented as a 3-tuple (uid, vid, key).
        """
        return list(self._graph.edges(keys=True))

    def successors(self, uid):
        """
        List of all successors of the node represented by uid.
        """
        return list(self._graph.successors(uid))

    def predecessors(self, uid):
        """
        List of all predecessors of the node represented by uid.
        """
        return list(self._graph.predecessors(uid))

    def neighbors(self, uid):
        """
        List of all (in and out) neighbors of the node represented by uid.
        """
        return list(self._graph.neighbors(uid))

    def ancestors(self, uid):
        """
        List of all nodes from which the node represented by uid is reachable.
        """
        return list(nx.ancestors(self._graph, uid))

    def descendants(self, uid):
        """
        List of all nodes that can be reached from  the node represented by uid.
        """
        return list(nx.descendants(self._graph, uid))

    def in_edges(self, uid):
        """
        List of all in edges to the node represented by uid.
        """
        return self._graph.in_edges(uid, keys=True)

    def is_isomorphic_to(self, other: 'Graph'):
        """
        Checks if the graph is isomorphic to the `other` graph.

        :param other: (:class:`Graph` object) Graph to be checked for isomorphism with current graph.
        :return: (bool) `True`, if graphs are isomorphic. Else, `False`.
        """
        return nx.is_isomorphic(self._graph, other._graph)

    def out_edges(self, uid):
        """
        List of all out edges from the node represented by uid.
        """
        return self._graph.out_edges(uid, keys=True)

    def number_of_nodes(self):
        """
        The number of nodes in the graph.
        """
        return self._graph.number_of_nodes()

    def number_of_edges(self):
        """
        The number of edges in the graph.
        """
        return self._graph.number_of_edges()

    def clear(self):
        """
        Clears all nodes, edges and the node, edge and graph properties.
        """
        self._graph.clear()
        self._node_properties = dict()
        self._edge_properties = dict()
        self._graph_properties = dict()

    def serialize(self):
        """
        Serializes the graph into a dictionary with the following format::

            {
                "graph": {
                    "nodes": <number of nodes>,
                    "edges": {
                        uid: {vid: key},
                        ...
                    }
                    "node_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": {
                                "uid": <property value>,
                                ...
                            }
                        },
                        ...
                    },
                    "edge_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": [{"edge": [uid, vid, key], "pvalue": <property value>} ...]
                        },
                        ...
                    },
                    "graph_properties": {
                        "property_name": <value>,
                        ...
                    }
                }
            }

        :return: (dict) Serialized graph
        """
        # Initialize a graph dictionary
        graph = dict()

        # Add nodes
        graph["nodes"] = self.number_of_nodes()

        # Add edges
        graph["edges"] = dict()
        for uid in range(self.number_of_nodes()):
            successors = list(self.successors(uid))
            if len(list(successors)) == 0:
                continue

            graph["edges"][uid] = dict()
            for vid in successors:
                graph["edges"][uid].update({vid: self._graph.number_of_edges(uid, vid)})

        # Add node properties
        # graph["node_properties"] = self._node_properties
        # graph["edge_properties"] = {
        #     prop_name: [
        #         {
        #             "edge": edge,
        #             "pvalue": pvalue
        #         }
        #         for edge, pvalue in prop_value.items()
        #     ]
        #     for prop_name, prop_value in self._edge_properties.items()
        # }
        graph["node_properties"] = {p_name: prop.serialize() for p_name, prop in self._node_properties.items()}
        graph["edge_properties"] = {p_name: prop.serialize() for p_name, prop in self._edge_properties.items()}
        graph["graph_properties"] = self._graph_properties

        # # Warn about any properties that were ignored.
        # ignored_attr = set(self.__dict__.keys()) - set(self._graph_properties.keys())
        # print(util.BColors.WARNING, f"[WARN] Attributes {ignored_attr} were not serialized because they are not "
        #                             f"node/edge/graph properties.", util.BColors.ENDC)

        # TODO. Add metadata such as time of serialization, serializer version etc.
        obj_dict = {"graph": graph}

        # Return serialized object
        return obj_dict

    @classmethod
    def deserialize(cls, obj_dict):
        """
        Constructs a graph from a serialized graph object. The format is described in :py:meth:`Graph.serialize`.

        :return: (Graph) A new :class:`Graph` object..
        """
        # Instantiate new object
        obj = cls()

        # Get serialized graph object
        graph_dict = obj_dict["graph"]

        # Add nodes
        obj.add_nodes(num_nodes=int(graph_dict["nodes"]))

        # Add edges
        edges = graph_dict["edges"]
        for uid in edges:
            for vid in edges[uid]:
                for key in range(edges[uid][vid]):
                    obj._graph.add_edge(int(uid), int(vid), key=int(key))

        # Add properties
        for node_prop, np_value in graph_dict["node_properties"].items():
            np_map = NodePropertyMap(graph=obj)
            # np_map.update({int(k): v for k, v in np_value.items()})
            np_map.deserialize(np_value)
            obj[node_prop] = np_map

        for graph_prop, gp_value in graph_dict["graph_properties"].items():
            obj[graph_prop] = gp_value

        for edge_prop, ep_value in graph_dict["edge_properties"].items():
            ep_map = EdgePropertyMap(graph=obj)
            ep_map.deserialize(ep_value)
            obj[edge_prop] = ep_map

        # Return constructed object
        return obj

    def save(self, fpath, overwrite=False, protocol="json", delimiter=";", remove_commas=False):
        """
        Saves the graph to file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param overwrite: (bool) Specifies whether to overwrite the file, if it exists. [Default: False]
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.
        """
        if not overwrite and os.path.exists(fpath):
            raise FileExistsError("File already exists. To overwrite, call Graph.save(..., overwrite=True).")

        graph_dict = self.serialize()
        if protocol == "json":
            with open(fpath, "w") as file:
                json.dump(graph_dict, file, indent=2)
        elif protocol == "pickle":
            with open(fpath, "wb") as file:
                pickle.dump(graph_dict, file)
        elif protocol == "adjacency_csv":
            # saves graph in an adjacency list csv format
            edge_dict = graph_dict["graph"]["edges"]
            with open(fpath, "w") as file:
                for uid in edge_dict:
                    file.write(f"{uid}")
                    vid_list = edge_dict[uid]
                    for vid in vid_list:
                        file.write(f"{delimiter}{vid}")
                    file.write("\n")
        elif protocol == "edgelist_csv":
            # save graph in an edge list csv format for support with cuGRAPH
            # https://medium.com/rapids-ai/large-graph-visualization-with-rapids-cugraph-590d07edce33
            # https: // docs.rapids.ai / api / cudf / stable / api_docs / api / cudf.read_csv.html
            edge_dict = graph_dict["graph"]["edges"]
            with open(fpath, "w") as file:
                file.write(f"source{delimiter}destination\n")
                for uid in edge_dict:
                    vid_list = edge_dict[uid]
                    for vid in vid_list:
                        file.write(f"{uid}{delimiter}{vid}\n")
        elif protocol == "metadata_csv":
            # save graph in a metadata format for use with cosmograph https://cosmograph.app/
            # user needs to add headers to the csv manually afterwards since the state can be in any format
            with open(fpath, "w") as file:
                for node in self.nodes():
                    file.write(f"{node}{delimiter}")
                    for state_variable in self._node_properties["state"][node]:
                        if remove_commas:
                            state_variable_cleaned = str(state_variable).replace(',', '')
                            file.write(f"{state_variable_cleaned}{delimiter}")
                        else:
                            file.write(f"{state_variable}{delimiter}")
                    file.write("\n")
        else:
            raise ValueError(f"Graph.save() does not support '{protocol}' protocol. One of ['json', 'pickle'] expected")

    @classmethod
    def load(cls, fpath, protocol="json"):
        """
        Loads the graph from file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError("File does not exist.")

        if protocol == "json":
            with open(fpath, "r") as file:
                obj_dict = json.load(file)
                graph = cls.deserialize(obj_dict)
        elif protocol == "pickle":
            with open(fpath, "rb") as file:
                obj_dict = pickle.load(file)
                graph = cls.deserialize(obj_dict)
        else:
            raise ValueError(f"Graph.load() does not support '{protocol}' protocol. One of ['json', 'pickle'] expected")

        return graph

    def to_png(self, fpath, nlabel=None, elabel=None):
        """
        Generates a PNG image of the graph.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param nlabel: (list of str) Specifies the node properties to use to annotate a node in image.
        :param elabel: (list of str) Specifies the edge properties to use to annotate an edge in image.

        :warning: If the node labels are not unique, the generated figure may contain 0, 1, 2, ...
            that avoid duplication.
        """
        max_nodes = 500
        if self._graph.number_of_nodes() > max_nodes:
            raise ValueError(f"Cannot draw a graph with more than {max_nodes} nodes.")

        g = self._graph

        # If node properties to displayed are specified, process them.
        if nlabel is not None:
            g = nx.MultiDiGraph()

            # If more than one property is selected, then display as tuple.
            if len(nlabel) == 1:
                node_state_map = {n: self[prop][n] for prop in nlabel for n in self._graph.nodes()}
            else:
                node_state_map = {n: tuple(self[prop][n] for prop in nlabel) for n in self._graph.nodes()}

            # Add nodes to dummy graph
            for n in node_state_map.values():
                g.add_node(str(n))

            # If edge labels to be displayed are specified, process them.
            if elabel is not None:
                for u, v, k in self._graph.edges(keys=True):
                    if len(elabel) == 1:
                        g.add_edge(str(node_state_map[u]), str(node_state_map[v]),
                                   label=self[elabel[0]][(u, v, k)])
                    else:
                        g.add_edge(str(node_state_map[u]), str(node_state_map[v]),
                                   label=tuple(self[prop][(u, v, k)] for prop in elabel))
            else:
                for u, v, k in self._graph.edges(keys=True):
                    g.add_edge(str(node_state_map[u]), str(node_state_map[v]))

        dot_graph = nx.nx_agraph.to_agraph(g)
        dot_graph.layout("dot")
        dot_graph.draw(fpath)

    def is_isomorphic_to(self, other: 'Graph'):
        """
        Checks if the graph is isomorphic to the `other` graph.

        :param other: (:class:`Graph` object) Graph to be checked for isomorphism with current graph.
        :return: (bool) `True`, if graphs are isomorphic. Else, `False`.
        """
        return nx.is_isomorphic(self._graph, other._graph)

    def base_graph(self):
        return self._graph

    def reverse(self):
        """
        Return a SubGraph.
        """
        return self._graph.reverse(copy=False)

    def bfs_layers(self, sources):
        return nx.bfs_layers(self._graph, sources)

    def reverse_bfs(self, sources):
        rev_graph = self._graph.reverse()
        reachable_nodes = set(reduce(set.union, list(map(set, nx.bfs_layers(rev_graph, sources)))))
        return reachable_nodes


class SubGraph(Graph):
    """
    A MultiDiGraph class represented as a 5-tuple (nodes, edges, node_properties, edge_properties, graph_properties).
    In addition, the graph implements a serialization protocol, save-load and drawing functionality.
    """
    def __init__(self, graph, hidden_nodes=None, hidden_edges=None):
        super(SubGraph, self).__init__()
        # Internal representation
        self._base_graph = graph
        self._graph = nx.subgraph_view(self._base_graph.base_graph(), self.is_node_visible, self.is_edge_visible)
        # self._hidden_nodes = set() if hidden_nodes is None else set(hidden_nodes)
        # self._hidden_edges = set() if hidden_edges is None else set(hidden_edges)
        self._hidden_nodes = NodePropertyMap(self._base_graph, default=False)
        self._hidden_edges = EdgePropertyMap(self._base_graph, default=False)
        self._base_graph["hidden_nodes"] = self._hidden_nodes
        self._base_graph["hidden_edges"] = self._hidden_edges

        # Initialize hidden nodes and edges
        if hidden_nodes is not None:
            for uid in hidden_nodes:
                self._hidden_nodes[uid] = True

        if hidden_edges is not None:
            for edge in hidden_edges:
                self._hidden_edges[edge] = True

        # Map node, edge and graph properties
        self._node_properties = graph._node_properties
        self._edge_properties = graph._edge_properties
        self._graph_properties = graph._graph_properties

    def __str__(self):
        return f"<SubGraph of {self._graph}>"

    def is_node_visible(self, uid):
        """
        Is the node included in the subgraph?
        """
        return not self._hidden_nodes[uid]

    def is_edge_visible(self, uid, vid, key):
        """
        Is the node included in the subgraph?
        """
        return not self._hidden_edges[(uid, vid, key)]

    def hide_node(self, uid):
        """
        Removes the node from subgraph.
        Raises error if `uid` is not in base graph.
        If `uid` was hidden then no change is made.
        """
        self._hidden_nodes[uid] = True

    def show_node(self, uid):
        """
        Adds the node back to subgraph.
        Raises error if `uid` is not in base graph.
        If `uid` was already visible, then no change is made.
        """
        self._hidden_nodes[uid] = False

    def hide_nodes(self, ulist):
        """
        Removes multiples nodes from subgraph.
        """
        map(self.hide_node, ulist)

    def show_nodes(self, ulist):
        """ Adds multiple nodes to subgraph. """
        map(self.show_node, ulist)

    def hidden_nodes(self):
        """ Gets the list of nodes in base graph that are not in subgraph. """
        return [uid for uid, value in self._hidden_nodes.items() if value is True]

    def visible_nodes(self):
        # return [uid for uid, value in self._hidden_nodes.items() if value is False]
        return list(set(self.nodes()) - set(self.hidden_nodes()))

    def number_of_visible_nodes(self):
        """ Gets the number of nodes in subgraph. """
        return self.number_of_nodes() - len(self.hidden_nodes())

    def hide_edge(self, uid, vid, key):
        """ Removes the edge from subgraph. No changes are made to base graph. """
        self._hidden_edges[(uid, vid, key)] = True

    def show_edge(self, uid, vid, key):
        """ Adds the edge to subgraph. The edge must be a valid edge in base graph. """
        self._hidden_edges[(uid, vid, key)] = False

    def hide_edges(self, elist):
        """ Removes multiple edge from subgraph. No changes are made to base graph. """
        map(self.hide_edge, elist)

    def show_edges(self, elist):
        """ Adds multiple edges to subgraph. The edge must be a valid edge in base graph. """
        map(self.show_edge, elist)

    def hidden_edges(self):
        """ Gets the list of edges from base graph that are not in subgraph. """
        return [edge for edge, value in self._hidden_edges.items() if value is True]

    def visible_edges(self):
        # return [edge for edge, value in self._hidden_edges.items() if value is False]
        return list(set(self.edges()) - set(self.hidden_edges()))

    def number_of_visible_edges(self):
        """ Gets the number of edges in subgraph. """
        return self.number_of_edges() - len(self.hidden_edges())

    def add_node(self):
        """
        Raises error. Nodes cannot be added to subgraph.
        See :meth:`SubGraph.hide_nodes` and :meth:`SubGraph.show_nodes`.
        """
        raise PermissionError("Cannot add nodes to a SubGraph.")

    def add_nodes(self, num_nodes):
        """
        Raises error. Nodes cannot be added to subgraph.
        See :meth:`SubGraph.hide_nodes` and :meth:`SubGraph.show_nodes`.
        """
        raise PermissionError("Cannot add nodes to a SubGraph.")

    def add_edge(self, uid, vid):
        """
        Raises error. Edges cannot be added to subgraph.
        See :meth:`SubGraph.hide_edges` and :meth:`SubGraph.show_edges`.
        """
        raise PermissionError("Cannot add edges to a SubGraph.")

    def add_edges(self, edges):
        """
        Raises error. Edges cannot be added to subgraph.
        See :meth:`SubGraph.hide_edges` and :meth:`SubGraph.show_edges`.
        """
        raise PermissionError("Cannot add edges to a SubGraph.")

    def rem_node(self, uid):
        """
        Raises error. Nodes cannot be removed to subgraph.
        See :meth:`SubGraph.hide_nodes` and :meth:`SubGraph.show_nodes`.
        """
        raise NotImplementedError("Removal of nodes is not supported. Use SubGraph.hide_node() instead.")

    def rem_edge(self, uid, vid, key):
        """
        Raises error. Edges cannot be removed to subgraph.
        See :meth:`SubGraph.hide_edges` and :meth:`SubGraph.show_edges`.
        """
        raise NotImplementedError("Removal of nodes is not supported. Use SubGraph instead.")

    def has_node(self, uid):
        """
        Checks whether the subgraph has the given node or not. Checks whether the node exists and is visible.

        :param uid: (int) Node ID to be checked for containment.
        :return: (bool) True if given node is in the graph, else False.
        """
        return self._graph.has_node(uid)

    def has_edge(self, uid, vid, key=None):
        """
        Checks whether the graph has the given edge or not. Checks whether the edge exists and is visible.

        :param uid: (int) Source node ID.
        :param vid: (int) Target node ID.
        :param key: If provided, checks whether the edge (u, v, k) is in the graph or not. Otherwise, checks if there
            exists an edge between nodes represented by uid and vid.
        :type key: int, optional
        :return: (bool) True if given edge is in the graph, else False.
        """
        return self._graph.has_edge(uid, vid, key)

    def nodes(self):
        """
        List of all nodes in the **subgraph**.
        """
        return list(self._graph.nodes())

    def edges(self):
        """
        List of all edges in the **subgraph**. Each edge is represented as a 3-tuple (uid, vid, key).
        """
        return list(self._graph.edges(keys=True))

    def successors(self, uid):
        """
        List of all successors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        return list(self._graph.successors(uid))

    def predecessors(self, uid):
        """
        List of all predecessors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        return list(self._graph.predecessors(uid))

    def neighbors(self, uid):
        """
        List of all (in and out) neighbors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        return list(self._graph.neighbors(uid))

    def ancestors(self, uid):
        """
        List of all nodes from which the node represented by uid is reachable.
        Includes only visible nodes reachable via visible edges.
        """
        return list(nx.ancestors(self._graph, uid))

    def descendants(self, uid):
        """
        List of all nodes that can be reached from  the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        return list(nx.descendants(self._graph, uid))

    def in_edges(self, uid):
        """
        List of all in edges to the node represented by uid.
        Includes only visible edges.
        """
        return self._graph.in_edges(uid, keys=True)

    def out_edges(self, uid):
        """
        List of all out edges from the node represented by uid.
        Includes only visible edges.
        """
        return self._graph.out_edges(uid, keys=True)

    def number_of_nodes(self):
        """
        The number of nodes in the **base** graph.
        """
        return self._graph.number_of_nodes()

    def number_of_edges(self):
        """
        The number of edges in the **base** graph.
        """
        return self._graph.number_of_edges()

    def clear(self):
        """
        Clears all nodes, edges and the node, edge and graph properties.

        .. warning:: The function is untested.
        # todo
        """
        self._graph.clear()
        self._node_properties = dict()
        self._edge_properties = dict()
        self._graph_properties = dict()

    def serialize(self):
        """
        Serializes the graph into a dictionary with the following format::

            {
                "graph": {
                    "nodes": <number of nodes>,
                    "edges": {
                        uid: {vid: key},
                        ...
                    }
                    "node_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": {
                                "uid": <property value>,
                                ...
                            }
                        },
                        ...
                    },
                    "edge_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": [{"edge": [uid, vid, key], "pvalue": <property value>} ...]
                        },
                        ...
                    },
                    "graph_properties": {
                        "property_name": <value>,
                        ...
                    }
                }
            }

        :return: (dict) Serialized graph
        """
        obj_dict = self._base_graph.serialize()
        return obj_dict

    @classmethod
    def deserialize(cls, obj_dict):
        """
        Constructs a graph from a serialized graph object. The format is described in :py:meth:`Graph.serialize`.

        :return: (Graph) A new :class:`Graph` object..

        .. warning:: The function is untested.
        # todo
        """
        # Instantiate new object
        obj = cls()

        # Get serialized graph object
        graph_dict = obj_dict["graph"]

        # Add nodes
        obj.add_nodes(num_nodes=int(graph_dict["nodes"]))

        # Add edges
        edges = graph_dict["edges"]
        for uid in edges:
            for vid in edges[uid]:
                for key in range(edges[uid][vid]):
                    obj._graph.add_edge(int(uid), int(vid), key=int(key))

        # Add properties
        for node_prop, np_value in graph_dict["node_properties"].items():
            np_map = NodePropertyMap(graph=obj)
            # np_map.update({int(k): v for k, v in np_value.items()})
            np_map.deserialize(np_value)
            obj[node_prop] = np_map

        for graph_prop, gp_value in graph_dict["graph_properties"].items():
            obj[graph_prop] = gp_value

        for edge_prop, ep_value in graph_dict["edge_properties"].items():
            ep_map = EdgePropertyMap(graph=obj)
            ep_map.deserialize(ep_value)
            obj[edge_prop] = ep_map

        # Return constructed object
        return obj

    def save(self, fpath, overwrite=False, protocol="json"):
        """
        Saves the graph to file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param overwrite: (bool) Specifies whether to overwrite the file, if it exists. [Default: False]
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.
        """
        if not overwrite and os.path.exists(fpath):
            raise FileExistsError("File already exists. To overwrite, call Graph.save(..., overwrite=True).")

        graph_dict = self.serialize()
        if protocol == "json":
            with open(fpath, "w") as file:
                json.dump(graph_dict, file, indent=2)
        elif protocol == "pickle":
            with open(fpath, "wb") as file:
                pickle.dump(graph_dict, file)
        else:
            raise ValueError(f"Graph.save() does not support '{protocol}' protocol. One of ['json', 'pickle'] expected")

    @classmethod
    def load(cls, fpath, protocol="json"):
        """
        Loads the graph from file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.

        .. warning:: The function is untested.
        # todo
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError("File does not exist.")

        if protocol == "json":
            with open(fpath, "r") as file:
                obj_dict = json.load(file)
                graph = cls.deserialize(obj_dict)
        elif protocol == "pickle":
            with open(fpath, "rb") as file:
                obj_dict = pickle.load(file)
                graph = cls.deserialize(obj_dict)
        else:
            raise ValueError(f"Graph.load() does not support '{protocol}' protocol. One of ['json', 'pickle'] expected")

        return graph

    def to_png(self, fpath, nlabel=None, elabel=None):
        """
        Generates a PNG image of the graph.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param nlabel: (list of str) Specifies the node properties to use to annotate a node in image.
        :param elabel: (list of str) Specifies the edge properties to use to annotate an edge in image.

        :warning: If the node labels are not unique, the generated figure may contain 0, 1, 2, ...
            that avoid duplication.

        .. warning:: The function is untested.
        # todo
        """
        max_nodes = 500
        if self._graph.number_of_nodes() > max_nodes:
            raise ValueError(f"Cannot draw a graph with more than {max_nodes} nodes.")

        g = self._graph

        # If node properties to displayed are specified, process them.
        if nlabel is not None:
            g = nx.MultiDiGraph()

            # If more than one property is selected, then display as tuple.
            if len(nlabel) == 1:
                node_state_map = {n: self[prop][n] for prop in nlabel for n in self._graph.nodes()}
            else:
                node_state_map = {n: tuple(self[prop][n] for prop in nlabel) for n in self._graph.nodes()}

            # Add nodes to dummy graph
            for n in node_state_map.values():
                g.add_node(str(n))

            # If edge labels to be displayed are specified, process them.
            if elabel is not None:
                for u, v, k in self._graph.edges(keys=True):
                    if len(elabel) == 1:
                        g.add_edge(str(node_state_map[u]), str(node_state_map[v]),
                                   label=self[elabel[0]][(u, v, k)])
                    else:
                        g.add_edge(str(node_state_map[u]), str(node_state_map[v]),
                                   label=tuple(self[prop][(u, v, k)] for prop in elabel))
            else:
                for u, v, k in self._graph.edges(keys=True):
                    g.add_edge(str(node_state_map[u]), str(node_state_map[v]))

        dot_graph = nx.nx_agraph.to_agraph(g)
        dot_graph.layout("dot")
        dot_graph.draw(fpath)

    def is_isomorphic_to(self, other: 'Graph'):
        """
        Checks if the graph is isomorphic to the `other` graph.

        :param other: (:class:`Graph` object) Graph to be checked for isomorphism with current graph.
        :return: (bool) `True`, if graphs are isomorphic. Else, `False`.

        .. warning:: The function is untested.
        # todo
        """
        return nx.is_isomorphic(self._graph, other._graph)
