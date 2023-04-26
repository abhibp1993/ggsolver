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
        Two PMaps are equal they store the same items and have the same defaults.

        .. note:: The two PMaps may have different names and be equal.
        """
        return self.default == other.default and set(self.items()) == set(other.items())

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

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        tmp = self.__class__(graph=self.graph).deserialize(obj_dict)
        self.__dict__.update(tmp.__dict__)

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
        if self.__class__.__name__ != obj_dict["type"]:
            raise TypeError(f"Cannot deserialize {obj_dict['type']=} into {self.__class__.__name__} class.")

        self.clear()
        self.default = obj_dict["default"]
        for k, v in obj_dict["map"].items():
            k = ast.literal_eval(k)
            if self.__contains__(k):
                self[k] = v


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


class PMapView(PMap):
    def __init__(self, graph, pmap):
        super(PMapView, self).__init__(graph=graph, default=pmap.default)
        self.pmap = pmap

    def __repr__(self):
        return f"<PMapView of {repr(self.pmap)} in {self.graph}>"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and super(PMapView, self).__eq__(other)

    def __contains__(self, item):
        return self.pmap.__contains__(item)

    def __getitem__(self, item):
        return self.pmap.__getitem__(item)

    def __setitem__(self, item, value):
        raise PermissionError(f"Cannot set value of property in {self.__class__.__name__}.")

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, obj_dict):
        pass

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
        serialized_dict["type"] = self.__class__.__name__
        return serialized_dict

    def deserialize(self, obj_dict):
        """
        Serialization of NPMView does not store any new information than the NodePropertyMap it views.
        Hence, no action is applied to `self` during deserialization.
        """
        return


class Graph:
    """
    Graph represents a 5-tuple

        $$G = (V, E, vp, edge_properties, graph_properties)$$

    where
        - $V$ is an index set $\{0, 1, ...\}$,
        - $E \subseteq V \times V \times \mathbb{N}$ is set of edges. Parallel edges are distinguished by a key.
        - $vp, edge_properties, graph_properties$ are node, edge and graph properties.

    The nodes and edges defined the underlying graph structure.

    Notes:
        - Each graph object is uniquely identified by its name.
    """
    _OBJECTS = dict()

    def __new__(cls, name=None, **kwargs):
        if name in Graph._OBJECTS:
            raise NameError(f"{cls.__name__} object with {name=} already exists. Cannot create duplicate.")

        # If name is not given, assign one. Pattern: Graph0, Graph1, ...
        if name is None:
            max_id = -1
            regex_pattern = r"\d+"
            for name in cls._OBJECTS.keys():
                match = re.search(regex_pattern, name)
                if match and int(match.group()) > max_id:
                    max_id = int(match.group())

            name = f"{cls.__name__}{max_id + 1}"

        # Create new object
        obj = super().__new__(cls)
        cls._OBJECTS[name] = obj
        return obj

    def __init__(self, name=None, **kwargs):
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
    # CLASS METHODS
    # ============================================================================================================
    @classmethod
    def get_instance_by_name(cls, name):
        return cls._OBJECTS[name]

    # ============================================================================================================
    # NODE/EDGE/GRAPH PROPERTY-RELATED METHODS
    # ============================================================================================================
    @property
    def node_properties(self):
        """ Returns the node properties as a dictionary of {"property name": NodePropertyMap object}. """
        return self._np

    @property
    def edge_properties(self):
        """ Returns the edge properties as a dictionary of {"property name": EdgePropertyMap object}. """
        return self._ep

    @property
    def graph_properties(self):
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

    def create_gp(self, pname, default=None, overwrite=False):
        if not overwrite:
            assert pname not in self._gp, f"Graph property: {pname} exists in graph:{self}. " \
                                          f"To overwrite pass parameter `overwrite=True` to this function."
        self._gp[pname] = default
        return

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
          "node_properties.<pname>": {
            "default": value,
            "map": {uid: value}
          },
          # For each edgeproperty
          "edge_properties.<pname>": {
            "default": value,
            "map": {eid: value}
          },
          # For each graph property
          "graph_properties.<pname>": value,
          # If type=subgraph then hierarchy information
          "hierarchy": {
            0: {    # The saved subgraph `SG`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.node_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.edge_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.graph_properties": <List[str]: names of properties that are not in default state>,
              "node_properties.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "edge_properties.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "graph_properties.<pname>": value,
            },
            1: {    # `SG.parent`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.node_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.edge_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.graph_properties": <List[str]: names of properties that are not in default state>,
              "node_properties.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "edge_properties.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "graph_properties.<pname>": value,
            },
            2: {    # `SG.parent.parent`
              "hidden_nodes": <List[int]>,
              "hidden_edges": <List[int]>,
              "modifiers.node_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.edge_properties": <List[str]: names of properties that are not in default state>,
              "modifiers.graph_properties": <List[str]: names of properties that are not in default state>,
              "node_properties.<pname>": {
                "default": value,
                "map": {uid: value}
              },
              "edge_properties.<pname>": {
                "default": value,
                "map": {eid: value}
              },
              "graph_properties.<pname>": value,
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
        graph["node_properties"] = list(self._np.keys())
        graph["edge_properties"] = list(self._ep.keys())
        graph["graph_properties"] = list(self._gp.keys())

        # Store properties
        for pname, pmap in self._np.items():
            graph["node_properties." + pname] = pmap.serialize()

        for pname, pmap in self._ep.items():
            graph["edge_properties." + pname] = pmap.serialize()

        for pname, pmap in self._gp.items():
            graph["graph_properties." + pname] = pmap

        # Return serialized object
        return graph

    def deserialize(self, obj_dict):
        """
        Constructs a graph from a serialized graph object. The format is described in :py:meth:`Graph.serialize`.
        :return: (Graph) A new :class:`Graph` object..
        """
        name = obj_dict["name"]
        if name in self.__class__._OBJECTS:
            logger.error(f"{self.__class__.__name__} with name:{name} exists. Skipped deserialization.")
            return self.__class__._OBJECTS[name]

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
        np = obj_dict["node_properties"]
        ep = obj_dict["edge_properties"]
        gp = obj_dict["graph_properties"]

        # Deserialize properties
        for pname in np:
            pmap = self.create_np(pname=pname)
            pmap.deserialize(obj_dict["node_properties." + pname])
            # graph[pname] = NodePMap(graph)
            # graph[pname].deserialize(obj_dict["node_properties." + pname])

        for pname in ep:
            pmap = self.create_ep(pname=pname)
            pmap.deserialize(obj_dict["edge_properties." + pname])
            # graph[pname] = EdgePMap(graph)
            # graph[pname].deserialize(obj_dict["edge_properties." + pname])

        for pname in gp:
            self[pname] = obj_dict["graph_properties." + pname]

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
    """
    * Memory management:
        - A subgraph SG shares nodes and edges (memory-wise) with the base graph G.
        - A subgraph SG shares node, edge, and graph properties (memory-wise) with the base graph G.

    * Users cannot modify graph topology in the sub-graph.
        That is, the addition or deletion of nodes and edges is prevented in SG.

    * By default, users may not modify node/edge/graph properties in SG. This is because SG stores PMapViews.
        There are two ways to modify the properties.
        1. Modify the property in the base graph.
            This may have unintended consequence of modifying the property for all the sub-graphs of the base graph.
        2. Modify the property locally, by flattening the property. Flattening replaces the PMapView with
            a NodePMap or EdgePMap which remain local to SG. Any changes to NodePMap or EdgePMap will not reflect
            to base graph.

    * It is possible to define a hierarchy of sub-graphs from a base graph. This includes
        - Defining SG2 = SubGraph(SG1), where SG1 = SubGraph(G).
        - Defining SG1 = SubGraph(G) and SG2 = SubGraph(G).

        In general, several subgraphs SG1, SG2, ... may have the same base graph G.
    * Each subgraph stores its parent graph/subgraph.

    * A subgraph is defined node and edge filters, that include a subset of nodes and edges of its parent.
        - This is stored in special private variable _visible_nodes, _visible_edges in the subgraph SG.
    """

    def __init__(self, parent, hidden_nodes=None, hidden_edges=None, name=None, **kwargs):
        """
        Constructs a subgraph given a parent :class:`Graph` or :class:`SubGraph`.

        :param parent: (:class:`Graph` or :class:`SubGraph`)
        :param hidden_nodes: (Iterable[int]) Iterable of node ids in parent (sub-)graph.
        :param hidden_edges: (Iterable[int, int, int]) Iterable of edge ids (uid, vid, key) in parent (sub-)graph.
        :param flatten_np: (Iterable[str] or "all") Iterable of node properties to be (shallow) copied.
            The remaining properties will be shared. If "all" then all properties are copied.
        :param flatten_ep: (Iterable[str] or "all") Iterable of edge properties to be (shallow) copied.
            The remaining properties will be shared. If "all" then all properties are copied.

        .. note:: Graph properties are always flattened.
            That is, any changes to graph_properties will remain local to SubGraph.
        """
        # Call to super
        super(SubGraph, self).__init__(name=name, **kwargs)

        # Assign default value to inputs, if not provided by user.
        hidden_nodes = set() if hidden_nodes is None else set(hidden_nodes)
        hidden_edges = set() if hidden_edges is None else set(hidden_edges)

        # Object representation
        # Note: `self._graph` which stores internal representation of Graph is set to None.
        #   This is to prevent SubGraph from inadvertently modifying the base graph structure.
        self._graph = None  # Set to None because super()._graph initializes to empty graph leading to silent bugs.
        self._parent = parent
        self._hidden_nodes = hidden_nodes   # self.create_np(pname="hidden_nodes", default=False)
        self._hidden_edges = hidden_edges   # self.create_ep(pname="hidden_edges", default=False)

        # # Initialize hidden nodes and hidden edges
        # if hidden_nodes is not None:
        #     for uid in hidden_nodes:
        #         self._hidden_nodes[uid] = True
        #
        # if hidden_edges is not None:
        #     for (uid, vid, key) in hidden_edges:
        #         self._hidden_edges[uid, vid, key] = True

        # Define node, edge and graph properties (store property map views)
        self._np = {
            k: PMapView(graph=self, pmap=v)
            for k, v in self._parent.node_properties.items()
        }
        self._ep = {
            k: PMapView(graph=self, pmap=v)
            for k, v in self._parent.edge_properties.items()
        }
        self._gp = self._parent.graph_properties.copy()

    def __repr__(self):
        return f"<SubGraph of {self._parent} with |V|={self.number_of_nodes()}, |E|={self.number_of_edges()}>"

    def __str__(self):
        if self.name is not None:
            return f"SubGraph({self._parent})"
        return self.__repr__()

    def __eq__(self, other: 'Graph'):
        # Equality of nodes
        if set(self.nodes()) != set(other.nodes()):
            return False

        # Equality of edges
        if set(self.edges()) != set(other.edges()):
            return False

        # Equality of properties
        return self._np == other._np and self._ep == other._ep and self._gp == other._gp

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
        # If pmap is either NodePMap or EdgePMap, update node_properties or edge_properties.
        if isinstance(pmap, PMap) and not isinstance(pmap, PMapView):
            if isinstance(pmap, NodePMap):
                pmap.name = pname
                self._np[pname] = pmap

            if isinstance(pmap, EdgePMap):
                pmap.name = pname
                self._ep[pname] = pmap

        elif isinstance(pmap, PMap):
            raise PermissionError(f"Cannot modify {pname} property of type {type(pmap)=} in sub-graph.")

        else:
            self._gp[pname] = pmap

    def __delitem__(self, pname, pmap):
        if isinstance(pmap, PMap) and not isinstance(pmap, PMapView):
            if pname in self._np.keys():
                self._np.pop(pname, None)

            if pname in self._ep.keys():
                self._ep.pop(pname, None)

        elif isinstance(pmap, PMap):
            raise PermissionError(f"Cannot delete {pname} property of type {type(pmap)=} in sub-graph.")

        else:
            self._gp.pop(pname, None)

    def __getstate__(self):
        parent = self.parent
        return {
                   "type": "SubGraph",
                   "ggsolver.version": version.ggsolver_version(),
                   "name": self.name,
                   "parent": parent,
                   "hidden_nodes": list(self._hidden_nodes),
                   "visible_nodes": list(self.nodes()),
                   "hidden_edges": list(self._hidden_edges),
                   "visible_edges": list(self.edges()),
                   "node_properties": list(self.node_properties.keys()),
                   "edge_properties": list(self.edge_properties.keys()),
                   "graph_properties": list(self.graph_properties.keys()),
               } | {
                   f"node_properties.{pname}": pmap.serialize() for pname, pmap in self.node_properties.items() if isinstance(pmap, NodePMap)
               } | {
                   f"edge_properties.{pname}": pmap.serialize() for pname, pmap in self.edge_properties.items() if isinstance(pmap, EdgePMap)
               } | {
                   f"graph_properties.{pname}": pmap for pname, pmap in self.graph_properties.items()
               }

    def __setstate__(self, obj_dict):
        self._parent = obj_dict["parent"]
        tmp = self.__class__(parent=self._parent).deserialize(obj_dict)
        self.__dict__.update(tmp.__dict__)

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

    # =====================================================================================
    # CLASS PROPERTIES
    # =====================================================================================
    @property
    def parent(self):
        return self._parent

    @property
    def base_graph(self):
        # If parent is a subgraph, return its base_graph.
        if issubclass(type(self.parent), SubGraph):
            return self.parent.base_graph

        # Else, parent is a Graph. Then return the parent.
        else:
            return self.parent

    # =====================================================================================
    # SUB-GRAPH SPECIAL METHODS
    # =====================================================================================
    def is_node_visible(self, uid):
        """
        Is the node included in the subgraph?
        If the subgraph is derived from another subgraph,
        then the node is visible if it is included in the subgraph, and it is visible in the parent.
        """
        return self.parent.has_node(uid) and uid not in self._hidden_nodes

    def is_edge_visible(self, uid, vid, key):
        """
        Is the edge included in the subgraph?
        If the subgraph is derived from another subgraph,
        then the edge is visible if it is included in the subgraph, and it is visible in the parent.
        """
        return self.parent.has_edge(uid, vid, key) and (uid, vid, key) not in self._hidden_nodes

    def hide_node(self, uid):
        """
        Removes the node from subgraph.
        Raises error if `uid` is not in base graph.
        If `uid` was hidden then no change is made.
        """
        # Hide in-edges
        for edge in self.parent.in_edges(uid):
            self.hide_edge(*edge)

        # Hide out-edges
        for edge in self.parent.out_edges(uid):
            self.hide_edge(*edge)

        # Hide node
        self._hidden_nodes.add(uid)

    def hide_nodes(self, ulist):
        """
        Hides multiple nodes from subgraph.
        """
        for uid in ulist:
            self.hide_node(uid)

    def show_node(self, uid):
        """
        Includes a hidden node into subgraph.
        Raises error if `uid` is not in base graph.
        If `uid` was already visible, then no change is made.
        """
        # Show node
        self._hidden_nodes.discard(uid)

        # Show in-edges
        for edge in self.parent.in_edges(uid):
            self.show_edge(*edge)

        # Hide out-edges
        for edge in self.parent.out_edges(uid):
            self.show_edge(*edge)

    def show_nodes(self, ulist):
        """
        Shows multiple nodes to subgraph.
        """
        for uid in ulist:
            self.show_node(uid)

    def hide_edge(self, uid, vid, key):
        """ Hides the edge from subgraph. No changes are made to base graph. """
        self._hidden_edges.add((uid, vid, key))

    def show_edge(self, uid, vid, key):
        """
        Shows the edge to subgraph. The edge must be a valid edge in base graph.
        .. note:: A hidden edge can be made "visible" only if uid and vid are both visible.
            Otherwise, no action is taken and a warning is issued.
        """
        if self.is_node_visible(uid) and self.is_node_visible(vid):
            self._hidden_edges.discard((uid, vid, key))
        else:
            logger.warning("fSubGraph.show_edge({uid, vid, key}) had no effect because source or target are hidden. "
                           f"See {self.is_node_visible(uid)=}, {self.is_node_visible(vid)=}")

    def hide_edges(self, elist):
        """ Hides multiple edge from subgraph. No changes are made to base graph. """
        for edge in elist:
            self.hide_edge(*edge)

    def show_edges(self, elist):
        """ Shows  multiple edges to subgraph. The edge must be a valid edge in base graph. """
        for edge in elist:
            self.show_edge(*edge)

    def hidden_nodes(self):
        """
        List of all hidden nodes in the **subgraph**.
        """
        return (uid for uid in self._hidden_nodes)

    def hidden_edges(self):
        """
        List of all hidden edges in the **subgraph**.
        """
        return (edge for edge in self._hidden_edges)

    def make_property_local(self, pname):
        """
        Flattens a property. That is, it creates a local copy of the node/edge property that can be modified by user.
        If property is not a PMapView instance, the function has no effect.
        :param pname:
        :return:
        """
        # If property is node/edge property, create a new property map and populate it.
        pmap = self[pname].copy()
        if pname in self.node_properties and isinstance(pmap, PMapView):
            pmap_flattened = self.create_np(pname=pname, default=pmap.default, overwrite=True)
            pmap_flattened.__dict__.update(pmap.__dict__)

        elif pname in self.edge_properties and isinstance(pmap, PMapView):
            pmap_flattened = self.create_ep(pname=pname, default=pmap.default, overwrite=True)
            pmap_flattened.__dict__.update(pmap.__dict__)

        else:
            logger.warning(f"{self}.make_property_local({pname}) had no effect.")

    # =====================================================================================
    # ACCESSING AND COUNTING NODES, EDGES
    # =====================================================================================
    def nodes(self):
        """
        List of all (visible) nodes in the **subgraph**.
        """
        return (uid for uid in self.parent.nodes() if uid not in self._hidden_nodes)

    def edges(self):
        """
        List of all (visible) edges in the **subgraph**.
        """
        return (edge for edge in self.parent.edges() if edge not in self._hidden_edges)

    def nodes_in_parent(self):
        """
        List of all nodes in the **subgraph's parent**.
        """
        return self.parent.nodes()

    def nodes_in_base_graph(self):
        """
        List of all nodes in the **base graph**.
        """
        return self.base_graph.nodes()

    def number_of_nodes(self):
        """ Gets the number of nodes in subgraph. """
        hidden_nodes = set(self._hidden_nodes)
        parent = self.parent
        while isinstance(parent, SubGraph):
            hidden_nodes.update(parent.hidden_nodes())
            parent = parent.parent

        return self.number_of_nodes_in_base_graph() - len(hidden_nodes)

    def number_of_nodes_in_parent(self):
        """ Gets the number of nodes in subgraph's parent. """
        return self.parent.number_of_nodes()

    def number_of_nodes_in_base_graph(self):
        """ Gets the number of nodes in subgraph's base graph. """
        return self.base_graph.number_of_nodes()

    def edges_in_parent(self):
        """
        List of all (visible) edges in the **subgraph's parent**.
        """
        return self.parent.edges()

    def edges_in_base_graph(self):
        """
        List of all edges in the **base graph**.
        """
        return self.base_graph.edges()

    def number_of_edges(self):
        """ Gets the number of edges in subgraph. """
        hidden_edges = set(self._hidden_edges)
        parent = self.parent
        while isinstance(parent, SubGraph):
            hidden_edges.update(parent.hidden_edges())
            parent = parent.parent

        return self.number_of_edges_in_base_graph() - len(hidden_edges)

    def number_of_edges_in_parent(self):
        """ Gets the number of edges in subgraph's parent. """
        return self.parent.number_of_edges()

    def number_of_edges_in_base_graph(self):
        """ Gets the number of edges in subgraph's base graph. """
        return self.base_graph.number_of_edges()

    # =====================================================================================
    # INHERITED GRAPH METHODS: BLOCK TOPOLOGY MODIFICATION
    # =====================================================================================
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

    def add_edge(self, uid, vid, key=None):
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
        raise PermissionError("Removal of nodes is not supported. Use SubGraph.hide_node() instead.")

    def rem_edge(self, uid, vid, key):
        """
        Raises error. Edges cannot be removed to subgraph.
        See :meth:`SubGraph.hide_edges` and :meth:`SubGraph.show_edges`.
        """
        raise PermissionError("Removal of nodes is not supported. Use SubGraph instead.")

    # =====================================================================================
    # CONTAINMENT CHECKING
    # =====================================================================================
    def has_node(self, uid):
        """
        Checks whether the subgraph has the given node or not. Checks whether the node exists and is visible.
        :param uid: (int) Node ID to be checked for containment.
        :return: (bool) True if given node is in the graph, else False.
        """
        return self.parent.has_node(uid) and uid not in self._hidden_nodes

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
        # The first edge added to graph has key=0.
        #   Hence, to check if there exists an edge between uid and vid, check if the edge (uid, vid, 0) is in graph.
        if key is None:
            key = 0

        return self.parent.has_edge(uid, vid, key) and (uid, vid, key) not in self._hidden_edges

    # =====================================================================================
    # NEIGHBORHOOD EXPLORATION
    # =====================================================================================
    def in_edges(self, uid):
        """
        List of all in edges to the node represented by uid.
        Includes only visible edges.
        """
        if not self.is_node_visible(uid):
            raise KeyError(f"{self.__class__.__name__}.in_edges({uid}):: "
                           f"Node is {'hidden' if self.base_graph.has_node(uid) else 'invalid'}")

        in_edges = self.parent.in_edges(uid)
        return ((uid, vid, key) for uid, vid, key in in_edges if self.is_edge_visible(uid, vid, key))

    def out_edges(self, uid):
        """
        List of all out edges from the node represented by uid.
        Includes only visible edges.
        """
        if not self.is_node_visible(uid):
            raise KeyError(f"{self.__class__.__name__}.in_edges({uid}):: "
                           f"Node is {'hidden' if self.base_graph.has_node(uid) else 'invalid'}")

        out_edges = self.parent.out_edges(uid)
        return ((uid, vid, key) for uid, vid, key in out_edges if self.is_edge_visible(uid, vid, key))

    def successors(self, uid):
        """
        List of all successors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        out_edges = self.out_edges(uid)
        return (v for u, v, k in out_edges)

    def predecessors(self, uid):
        """
        List of all predecessors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        in_edges = self.in_edges(uid)
        return (u for u, v, k in in_edges)

    def neighbors(self, uid):
        """
        List of all (in and out) neighbors of the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        yield from self.successors(uid)
        yield from self.predecessors(uid)

    def ancestors(self, uid):
        """
        List of all nodes from which the node represented by uid is reachable.
        Includes only visible nodes reachable via visible edges.
        """
        if not self.is_node_visible(uid):
            raise KeyError(f"{self.__class__.__name__}.in_edges({uid}):: "
                           f"Node is {'hidden' if self.base_graph.has_node(uid) else 'invalid'}")

        queue = [uid]
        visited = {uid}
        while queue:
            node = queue.pop()
            for neighbor in self.predecessors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def descendants(self, uid):
        """
        List of all nodes that can be reached from the node represented by uid.
        Includes only visible nodes reachable via visible edges.
        """
        if not self.is_node_visible(uid):
            raise KeyError(f"{self.__class__.__name__}.in_edges({uid}):: "
                           f"Node is {'hidden' if self.base_graph.has_node(uid) else 'invalid'}")

        queue = [uid]
        visited = {uid}
        while queue:
            node = queue.pop()
            for neighbor in self.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def bfs(self, sources, edges=False):
        """
        Traverse subgraph in breadth-first manner.
        :param sources:
        :param edges:
        :return: (List[nodes], List[edges]).
        """
        raise NotImplementedError("Request feature.")

    def dfs(self, sources, edges=False):
        """
        Traverse subgraph in breadth-first manner.
        :param sources:
        :param edges:
        :return: (List[nodes], List[edges]).
        """
        raise NotImplementedError("Request feature.")

    def is_isomorphic_to(self, other: 'Graph'):
        """
        Checks if the graph is isomorphic to the `other` graph.
        :param other: (:class:`Graph` object) Graph to be checked for isomorphism with current graph.
        :return: (bool) `True`, if graphs are isomorphic. Else, `False`.
        """
        raise NotImplementedError("Request feature")

    # =====================================================================================
    # MISCELLANEOUS METHODS
    # =====================================================================================
    def clear(self):
        """
        Resets the modified node, edge and graph properties.
        No changes done to hidden nodes, hidden edges.
        """
        self._np = {
            k: PMapView(graph=self, pmap=v)
            for k, v in self.parent.node_properties.items()
        }
        self._ep = {
            k: PMapView(graph=self, pmap=v)
            for k, v in self.parent.edge_properties.items()
        }
        self._gp = self.parent.graph_properties.copy()

    # =====================================================================================
    # SERIALIZATION
    # =====================================================================================
    def serialize(self):
        # base_graph = self.base_graph.serialize()
        parent = self.parent.serialize()
        return {
            "type": "SubGraph",
            "ggsolver.version": version.ggsolver_version(),
            "name": self.name,
            "parent": parent,
            "hidden_nodes": list(self._hidden_nodes),
            "hidden_edges": list(self._hidden_edges),
            "node_properties": [pname for pname, pmap in self.node_properties.items() if isinstance(pmap, NodePMap)],
            "edge_properties": [pname for pname, pmap in self.edge_properties.items() if isinstance(pmap, EdgePMap)],
            "graph_properties": list(self.graph_properties.keys()),
        } | {
            f"node_properties.{pname}": pmap.serialize() for pname, pmap in self.node_properties.items() if isinstance(pmap, NodePMap)
        } | {
            f"edge_properties.{pname}": pmap.serialize() for pname, pmap in self.edge_properties.items() if isinstance(pmap, EdgePMap)
        } | {
            f"graph_properties.{pname}": pmap for pname, pmap in self.graph_properties.items()
        }

    def deserialize(self, obj_dict):
        """
        Assume the SubGraph parent is already constructed.

        :param obj_dict:
        :return:
        """
        # Clear graph
        self.clear()

        # Process metadata
        if obj_dict["type"] != "SubGraph":
            raise TypeError(f"Cannot construct {self.__class__.__name__} object from {obj_dict['type']}.")

        obj_version = obj_dict["ggsolver.version"]
        obj_version = [int(part) for part in obj_version.split('.')]
        curr_version = [int(part) for part in version.ggsolver_version().split('.')]
        if obj_version[0] < curr_version[0] or obj_version[1] < curr_version[1]:
            logger.warning(f"Attempting to deserialize SubGraph saved in {obj_version} in "
                           f"ggsolver ver. {version.ggsolver_version()} may lead to unexpected issues.")

        # Update name
        self.name = obj_dict["name"]

        # Load hidden nodes and hidden edges
        self._hidden_nodes = set(obj_dict["hidden_nodes"])
        self._hidden_edges = set(obj_dict["hidden_edges"])

        # Validate hidden nodes and edges
        if any(not self.base_graph.has_node(uid) for uid in self._hidden_nodes) or \
                any(not self.base_graph.has_edge(*edge) for edge in self._hidden_edges):
            raise RuntimeError(f"Subgraph could not be deserialized. "
                               f"{self.parent} does not have all hidden nodes or edges from the serialized graph.")

        # Property metadata
        np = obj_dict["node_properties"]
        ep = obj_dict["edge_properties"]
        gp = obj_dict["graph_properties"]

        # Deserialize properties
        for pname in np:
            if "node_properties." + pname in obj_dict:
                pmap = self.create_np(pname=pname, overwrite=True)
                pmap.deserialize(obj_dict["node_properties." + pname])

        for pname in ep:
            if "edge_properties." + pname in obj_dict:
                pmap = self.create_ep(pname=pname, overwrite=True)
                pmap.deserialize(obj_dict["edge_properties." + pname])

        self.graph_properties.clear()
        for pname in gp:
            self.graph_properties[pname] = obj_dict["graph_properties." + pname]

        return self
