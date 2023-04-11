"""
Notes:
    * Graph is implemented with NetworkX backend.
"""
import ast
import itertools
import pathlib
import networkx as nx
import ggsolver.version as version
import ggsolver.ioutils as io
from loguru import logger
from functools import reduce


class PMap(dict):
    """ Base class for NodePropertyMap and EdgePropertyMap. """
    def __init__(self, graph, pname=None, default=None):
        super(PMap, self).__init__()
        self.pname = pname
        self.graph = graph
        self.default = default

    def __repr__(self):
        if self.pname is None:
            return f"<{self.__class__.__name__} of {repr(self.graph)}>"
        return f"<{self.__class__.__name__} of {repr(self.graph)}.{self.pname}>"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """
        Two PMaps are equal if they are defined over same graph and store the same default and items.

        .. note:: The two PMaps may have different names and be equal.
        """
        return self.graph == other.graph and self.default == other.default and set(self.items()) == set(other.items())

    __hash__ = object.__hash__

    def __contains__(self, item):
        raise NotImplementedError("Abstract.")

    def __missing__(self, item):
        # TODO. Replace with `if item in self.graph` after Graph.__contains__ is checked.
        if self.__contains__(item):
            return self.default
        raise KeyError(f"{self.__class__.__name__}.__missing__:: {item} is not in {self.graph}.")

    def __getitem__(self, item):
        if not self.__contains__(item):
            raise KeyError(f"[ERROR] {self.__class__.__name__}.__getitem__:: {item} is not in {self.graph}.")

        try:
            return super(PMap, self).__getitem__(item)
        except KeyError:
            return self.__missing__(item)

    def __setitem__(self, item, value):
        if not self.__contains__(item):
            raise KeyError(f"{self.__class__.__name__}.__setitem__:: {item} not in {self.graph}.")
        if value != self.default:
            super(PMap, self).__setitem__(item, value)

    def keys(self):
        """
        Returns all valid keys in property map. Typically, this includes all graph nodes or edges.
        """
        raise NotImplementedError("Abstract.")

    def items(self):
        """
        Returns all valid items in property map. Typically, this includes all graph nodes or edges.
        """
        return ((k, super(self.__class__, self).__getitem__(k)) for k in self.keys())

    def local_keys(self):
        """
        Returns only the keys that are explicitly stored in property map.
        """
        return super(PMap, self).keys()

    def local_items(self):
        """
        Returns only the items that are explicitly stored in property map.
        """
        return super(PMap, self).items()

    def update_default(self, new_default):
        """
        Updates the default value. This is linear time computation in total number of keys given by `self.keys()`.
        """
        old_default = self.default
        for k, v in self.items():
            if v == old_default:
                super(PMap, self).__setitem__(k, v)

            if v == new_default:
                super(PMap, self).pop(k)

    def serialize(self):
        # Construct a map of non-default values.
        non_default_items = {str(k): v for k, v in self.local_items()}

        return {
            "type": self.__class__.__name__,
            "default": self.default,
            "map": non_default_items
        }

    def deserialize(self, obj_dict):
        self.clear()
        self.default = obj_dict["default"]
        for k, v in obj_dict["map"].items():
            if self.__contains__(k):
                self[ast.literal_eval(k)] = v


class PMapView(PMap):
    def __init__(self, graph, pmap):
        super(PMapView, self).__init__(graph=graph, default=pmap.default)
        self.pmap = pmap

    def __repr__(self):
        return f"<PMapView of {repr(self.pmap)} in {self.graph}>"

    def __str__(self):
        return self.__repr__()

    def __contains__(self, item):
        return self.pmap.__contains__(item)

    def __getitem__(self, item):
        return self.pmap.__getitem__(item)

    def __setitem__(self, item, value):
        raise PermissionError(f"Cannot set value of property in {self.__class__.__name__}.")

    def keys(self):
        return self.pmap.keys()

    def items(self):
        return self.pmap.items()

    def local_keys(self):
        """ For a PMView, no keys are stored locally. Return all keys of the property map "self" views. """
        return self.pmap.keys()

    def local_items(self):
        """ For a PMView, no keys are stored locally. Return all items() of the property map "self" views. """
        return self.pmap.items()

    def update_default(self, new_default):
        raise PermissionError(f"Cannot update default of {self.__class__.__name__}.")

    def serialize(self):
        # Return serialization of underlying PMap.
        serialized_dict = self.pmap.serialize()
        return serialized_dict

    def deserialize(self, obj_dict):
        """
        Serialization of NPMView does not store any new information than the NodePropertyMap it views.
        Hence, no action is applied to `self` during deserialization.
        """
        return


class NodePMap(PMap):
    """
    Implements a default dictionary that maps a node ID to its property value. To store data efficiently,
    only the non-default values are stored in the dictionary.
    Raises an error if the node ID is invalid.
    """
    def __contains__(self, item):
        # TODO. Can be replaced with `return item in self.graph`.
        return self.graph.has_node(item)

    def keys(self):
        return self.graph.nodes()


class EdgePMap(PMap):
    """
    Implements a default dictionary that maps an edge (uid, vid, key) to its property value. To store data efficiently,
    only the non-default values are stored in the dictionary.
    Raises an error if the edge (uid, vid, key) is invalid.
    """

    def __contains__(self, item):
        # TODO. Can be replaced with `return item in self.graph`.
        return self.graph.has_edge(*item)

    def keys(self):
        return self.graph.edges()


class Graph:
    """
    Graph represents a 5-tuple

        $$G = (V, E, vp, ep, gp)$$

    where
        - $V$ is an index set $\{0, 1, ...\}$,
        - $E \subseteq V \times V \times \mathbb{N}$ is set of edges. Parallel edges are distinguished by a key.
        - $vp, ep, gp$ are node, edge and graph properties.

    The nodes and edges defined the underlying graph structure.

    Notes:
        - Each graph object is uniquely identified by its name.
    """
    def __init__(self, name=None):
        self.name = name
        self._graph = nx.MultiDiGraph()
        self._np = dict()
        self._ep = dict()
        self._gp = dict()

    def __repr__(self):
        return f"<Graph with |V|={self.number_of_nodes()}, |E|={self.number_of_edges()}>"

    def __str__(self):
        if self.name is not None:
            return f"Graph({self.name})"
        return self.__repr__()

    def __getitem__(self, pname):
        if pname in self._np:
            return self._np[pname]
        elif pname in self._ep:
            return self._ep[pname]
        elif pname in self._gp:
            return self._gp[pname]
        else:
            raise KeyError(f"{pname} is not a valid node/edge/graph property.")

    def __setitem__(self, pname, pmap):
        # Check if the property exists
        if pname in self._np.keys():
            assert isinstance(pmap, NodePMap)

        if pname in self._ep.keys():
            assert isinstance(pmap, EdgePMap)

        # Update property value
        if isinstance(pmap, NodePMap):
            pmap.name = pname
            self._np[pname] = pmap
        elif isinstance(pmap, EdgePMap):
            pmap.name = pname
            self._ep[pname] = pmap
        else:
            self._gp[pname] = pmap

    def __delitem__(self, pname, pmap):
        if pname in self._np.keys():
            self._np.pop(pname, None)

        if pname in self._ep.keys():
            self._ep.pop(pname, None)

        self._gp.pop(pname, None)

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        obj = self.__class__().deserialize(obj_dict)
        self.__dict__.update(obj.__dict__)

    def __eq__(self, other: 'Graph'):
        # Equality of nodes
        if self.number_of_nodes() != other.number_of_nodes():
            return False

        # Equality of edges
        self_edges = set(self.edges())
        other_edges = set(other.edges())
        if self_edges != other_edges:
            return False

        # Equality of properties
        return self._np == other._np and self._ep == other._ep and self._gp == other._gp

    def __len__(self):
        return self.number_of_nodes()

    def __contains__(self, item):
        if isinstance(item, int):
            return self.has_node(item)
        elif len(item) == 3 and all(isinstance(i, int) for i in item):
            return self.has_edge(*item)
        else:
            raise TypeError(f"{item=} must be either an int (to check `node` in graph) or a "
                            f"3-tuple of int (to check `edge` in graph). ")

    __hash__ = object.__hash__

    # ============================================================================================================
    # NODE/EDGE/GRAPH PROPERTY-RELATED METHODS
    # ============================================================================================================
    @property
    def np(self):
        """ Returns the node properties as a dictionary of {"property name": NodePropertyMap object}. """
        return self._np

    @property
    def ep(self):
        """ Returns the edge properties as a dictionary of {"property name": EdgePropertyMap object}. """
        return self._ep

    @property
    def gp(self):
        """ Returns the graph properties as a dictionary of {"property name": property value}. """
        return self._gp

    def create_np(self, pname, default=None, overwrite=False):
        if not overwrite:
            assert pname not in self._np, f"Node property: {pname} exists in graph:{self}. " \
                                          f"To overwrite pass parameter `overwrite=True` to this function."
        np = NodePMap(graph=self, pname=pname, default=default)
        self[pname] = np
        return np

    def create_ep(self, pname, default=None, overwrite=False):
        if not overwrite:
            assert pname not in self._ep, f"Edge property: {pname} exists in graph:{self}. " \
                                          f"To overwrite pass parameter `overwrite=True` to this function."
        ep = EdgePMap(graph=self, pname=pname, default=default)
        self[pname] = ep
        return ep

    # ============================================================================================================
    # GRAPH TOPOLOGY MANIPULATION METHODS
    # ============================================================================================================
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
        :return: (generator) IDs of added nodes.
        """
        return list(self.add_node() for _ in range(num_nodes))

    def add_edge(self, uid, vid):
        """
        Adds a new edge between the give nodes.
        :warning: Duplication is NOT checked. Hence, calling the function twice adds two parallel edges between
            the same nodes.
        :return: (int) Key of the added edge. Key = 0 means the first edge was added between the given nodes.
            If Key = k, then (k+1)-th edge was added.
        """
        if not self.__contains__(uid) or not self.__contains__(vid):
            raise KeyError(f"{self.__class__.__name__}.add_edge:: Adding edge from {uid=} to {vid=} failed."
                           f"{not self.__contains__(uid)=}, {not self.__contains__(vid)=}")
        return self._graph.add_edge(uid, vid)

    def add_edges(self, edges):
        """
        Adds multiple edge between the give nodes.
        :warning: Duplication is NOT checked. Hence, calling the function twice adds two parallel edges between
            the same nodes.
        :return: (int) Key of the added edge. Key = 0 means the first edge was added between the given nodes.
            If Key = k, then (k+1)-th edge was added.
        """
        return list(self.add_edge(*edge) for edge in edges)

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
        Generator over nodes in the graph.
        """
        return (uid for uid in self._graph.nodes())

    def edges(self):
        """
        Generator of all edges in the graph. Each edge is represented as a 3-tuple (uid, vid, key).
        """
        return (edge for edge in self._graph.edges(keys=True))

    def successors(self, uid):
        """
        Generator of all successors of the node represented by uid.
        """
        return (uid for uid in self._graph.successors(uid))

    def predecessors(self, uid):
        """
        Iterator of all predecessors of the node represented by uid.
        """
        return (uid for uid in self._graph.predecessors(uid))

    def neighbors(self, uid):
        """
        Generator of all (in and out) neighbors of the node represented by uid.
        """
        return (uid for uid in itertools.chain(self.successors(uid), self.predecessors(uid)))

    def ancestors(self, uid):
        """
        Generator of all nodes from which the node represented by uid is reachable.
        """
        return (uid for uid in nx.ancestors(self._graph, uid))

    def descendants(self, uid):
        """
        Generator of all nodes that can be reached from  the node represented by uid.
        """
        return (uid for uid in nx.descendants(self._graph, uid))

    def in_edges(self, uid):
        """
        Generator of all in edges to the node represented by uid.
        """
        return (edge for edge in self._graph.in_edges(uid, keys=True))

    def out_edges(self, uid):
        """
        List of all out edges from the node represented by uid.
        """
        return (edge for edge in self._graph.out_edges(uid, keys=True))

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

    def is_isomorphic_to(self, other: 'Graph'):
        """
        Checks if the graph is isomorphic to the `other` graph.
        :param other: (:class:`Graph` object) Graph to be checked for isomorphism with current graph.
        :return: (bool) `True`, if graphs are isomorphic. Else, `False`.
        """
        return nx.is_isomorphic(self._graph, other._graph)

    def graph_repr(self):
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

    def cycles(self):
        return nx.simple_cycles(self._graph)

    def clear(self):
        """
        Clears all nodes, edges and the node, edge and graph properties.
        """
        self.name = None
        self._graph.clear()
        self._np = dict()
        self._ep = dict()
        self._gp = dict()

    # =================================================================================================================
    # SERIALIZATION AND DESERIALIZATION
    # =================================================================================================================
    def serialize(self):
        """
        Serializes the graph into a dictionary with the following format::
        ```json
        {
          # Metadata
          "type": <Graph or SubGraph>,
          # Graph topology
          "nodes": <int>,
          "edges": <List[int]>, Use cantor mapping for (uid, vid, key) <-> eid
          # Graph property metadata
          "node_properties": <List[str]>,
          "edge_properties": <List[str]>,
          "graph_properties": <List[str]>,
          # For each node property
          "np.<pname>": {
            "default": value,
            "map": {uid: value}
          },
          # For each edgeproperty
          "ep.<pname>": {
            "default": value,
            "map": {eid: value}
          },
          # For each graph property
          "gp.<pname>": value,
          # If type=subgraph then hierarchy information
          "hierarchy": {
            0: {    # The saved subgraph `SG`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.np": <List[str]: names of properties that are not in default state>,
              "modifiers.ep": <List[str]: names of properties that are not in default state>,
              "modifiers.gp": <List[str]: names of properties that are not in default state>,
              "np.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "ep.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "gp.<pname>": value,
            },
            1: {    # `SG.parent`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.np": <List[str]: names of properties that are not in default state>,
              "modifiers.ep": <List[str]: names of properties that are not in default state>,
              "modifiers.gp": <List[str]: names of properties that are not in default state>,
              "np.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "ep.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "gp.<pname>": value,
            },
            2: {    # `SG.parent.parent`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.np": <List[str]: names of properties that are not in default state>,
              "modifiers.ep": <List[str]: names of properties that are not in default state>,
              "modifiers.gp": <List[str]: names of properties that are not in default state>,
              "np.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "ep.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "gp.<pname>": value,
            }
          }
        }
        ```
        :return: (dict) Serialized graph
        """
        # Initialize a graph dictionary
        graph = dict()

        # Metadata
        graph["type"] = "Graph"
        graph["ggsolver.version"] = version.ggsolver_version()
        graph["name"] = self.name

        # Topology
        graph["nodes"] = self.number_of_nodes()
        graph["edges"] = [str(edge) for edge in self.edges()]

        # Property metadata
        graph["np"] = list(self._np.keys())
        graph["ep"] = list(self._ep.keys())
        graph["gp"] = list(self._gp.keys())

        # Store properties
        for pname, pmap in self._np.items():
            graph["np." + pname] = pmap.serialize()

        for pname, pmap in self._ep.items():
            graph["ep." + pname] = pmap.serialize()

        for pname, pmap in self._gp.items():
            graph["gp." + pname] = pmap.serialize()

        # Return serialized object
        return graph

    def deserialize(self, obj_dict):
        """
        Constructs a graph from a serialized graph object. The format is described in :py:meth:`Graph.serialize`.
        :return: (Graph) A new :class:`Graph` object..
        """
        # Clear graph
        self.clear()

        # Process metadata
        if obj_dict["type"] != "Graph":
            raise TypeError(f"Cannot construct {self.__class__.__name__} object from {obj_dict['type']}.")

        obj_version = obj_dict["ggsolver.version"]
        obj_version = [int(part) for part in obj_version.split('.')]
        curr_version = [int(part) for part in version.ggsolver_version().split('.')]
        if obj_version[0] < curr_version[0] or obj_version[1] < curr_version[1]:
            logger.warning(f"Attempting to deserialize Graph saved in {obj_version} in "
                           f"ggsolver ver. {version.ggsolver_version()} may lead to unexpected issues.")

        # Update name
        self.name = obj_dict["name"]

        # Add nodes
        self.add_nodes(num_nodes=int(obj_dict["nodes"]))

        # Add edges
        edges = (ast.literal_eval(eid) for eid in obj_dict["edges"])
        self._graph.add_edges_from(edges)

        # Property metadata
        np = obj_dict["np"]
        ep = obj_dict["ep"]
        gp = obj_dict["gp"]

        # Deserialize properties
        for pname in np:
            pmap = self.create_np(pname=pname)
            pmap.deserialize(obj_dict["np." + pname])
            # graph[pname] = NodePMap(graph)
            # graph[pname].deserialize(obj_dict["np." + pname])

        for pname in ep:
            pmap = self.create_ep(pname=pname)
            pmap.deserialize(obj_dict["ep." + pname])
            # graph[pname] = EdgePMap(graph)
            # graph[pname].deserialize(obj_dict["ep." + pname])

        for pname in gp:
            self[pname] = obj_dict["gp." + pname]

        # Return constructed object
        return self

    # =================================================================================================================
    # SAVE AND LOAD
    # =================================================================================================================
    def save(self, fpath, protocol="json", overwrite=False):
        if not overwrite and pathlib.Path(fpath).exists():
            raise FileExistsError("File already exists. To overwrite, call Graph.save(..., overwrite=True).")

        graph_dict = self.serialize()
        if protocol == "json":
            io.to_json(fpath, graph_dict)
        elif protocol == "pickle":
            io.to_pickle(fpath, graph_dict)
        elif protocol == "dot":
            io.to_dot(fpath, self)
        elif protocol == "cosmograph":
            io.to_cosmograph(fpath, self)
        else:
            raise TypeError(f"Graph.save() does not support {protocol=}.")

    def load(self, fpath, protocol="json"):
        if protocol == "json":
            obj_dict = io.from_json(fpath)
        elif protocol == "pickle":
            obj_dict = io.from_pickle(fpath)
        elif protocol == "dot":
            obj_dict = io.from_dot(fpath, self)
        elif protocol == "cosmograph":
            raise TypeError(f"Graph.load() supports only saving to cosmograph. Cannot load from cosmograph files.")
        else:
            raise TypeError(f"Graph.load() does not support {protocol=}.")

        return self.deserialize(obj_dict)


class SubGraph(Graph):
    pass
