"""
Commands:
- `python3 -m ggsolver graphify <opts> model
"""
import argparse
import importlib.util
import os
from datetime import datetime
from prettytable import PrettyTable

import ggsolver
import ggsolver.dtptb as dtptb
import ggsolver.ioutils as io


def cmd_graphify(args):
    spec = importlib.util.spec_from_file_location("graphify", args.model)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    if args.list is True:
        # Create a new table object with headers
        table = PrettyTable()
        table.field_names = ["Object Name", "Object Type"]

        # Add rows to the table from the dictionary
        for name, obj in model.__dict__.items():
            if isinstance(obj, ggsolver.Game):
                table.add_row([name, obj])

        # Set the table style
        table.align = 'l'
        print(table)

    else:  # Run graphify on the model
        game = model.__dict__[args.game]
        graph = game.graphify(
            pointed=args.pointed,
            cores=1,
            verbosity=args.verbosity,
            np=list(args.np),
            ep=list(args.ep),
            gp=list(args.gp),
            ignore_np=list(args.ignore_np),
            ignore_ep=list(args.ignore_ep),
            ignore_gp=list(args.ignore_gp),
        )
        if args.out is None:
            print(graph.serialize())
        else:
            graph.save(args.out, protocol=args.protocol, overwrite=args.r)


def cmd_dtptb(args):
    graph = ggsolver.Graph().load(args.graph, protocol=args.protocol)
    if str(args.solver).lower() == "swinreach":
        swin = dtptb.SWinReach(
            graph=graph,
            solver=args.backend,
            player=args.player,
            verbosity=args.verbosity,
            directory=args.directory,
            filename=args.filename
        )
        swin.solve()
        solution = swin.solution()
        solution.save(os.path.join(args.directory, f"{args.filename}.solution"))


def cmd_dotviz(args):
    if args.style in ["simple", "solution"]:
        graph = ggsolver.Graph().load(args.inp)
        io.to_dot(
            graph=graph,
            fpath=args.out,
            formatting=args.style,
            node_props=args.np,
            edge_props=args.ep
        )
    else:
        raise NotImplementedError("Currently automaton visualization is not implemented.")


def configure_cmd_graphify(subparser):
    # create the parser for the 'graphify' command
    parser_ggsolver_solvers = subparser.add_parser("graphify")
    parser_ggsolver_solvers.add_argument("model", type=str,
                                         help="Python file containing game model definition")
    parser_ggsolver_solvers.add_argument("-l", "--list", action="store_true", default=False,
                                         help="Lists all Game objects in given python file. Must be in globals()")
    parser_ggsolver_solvers.add_argument("-g", "--game", type=str, default=None,
                                         help="Name of game object to graphify")
    parser_ggsolver_solvers.add_argument("-p", "--pointed", action="store_true", default=False,
                                         help="Use pointed graphify. Expects the game to be initialized.")
    # parser_ggsolver_solvers.add_argument("-c", "--cores", type=int, default=1,
    #                                      help="Number of cores to use for graphify.")
    parser_ggsolver_solvers.add_argument("-v", "--verbosity", type=int, default=0,
                                         help="Verbosity level to use for graphify.")
    parser_ggsolver_solvers.add_argument("-np", nargs="+", default=list(),
                                         help="List of node properties to include in graph.")
    parser_ggsolver_solvers.add_argument("-ep", nargs="+", default=list(),
                                         help="List of edge properties to include in graph.")
    parser_ggsolver_solvers.add_argument("-gp", nargs="+", default=list(),
                                         help="List of graph properties to include in graph.")
    parser_ggsolver_solvers.add_argument("-inp", "--ignore-np", nargs="+", default=list(),
                                         help="List of node properties to ignore in graph.")
    parser_ggsolver_solvers.add_argument("-iep", "--ignore-ep", nargs="+", default=list(),
                                         help="List of edge properties to ignore in graph.")
    parser_ggsolver_solvers.add_argument("-igp", "--ignore-gp", nargs="+", default=list(),
                                         help="List of graph properties to ignore in graph.")
    parser_ggsolver_solvers.add_argument("-o", "--out", type=str, default=None,
                                         help="Output file path (with extension) to store generated graph.")
    parser_ggsolver_solvers.add_argument("-k", "--protocol", type=str, default="json",
                                         help="Protocol to save generated graph.")
    parser_ggsolver_solvers.add_argument("-r", action="store_true", default=False,
                                         help="Overwrite the output file, if exists.")


def configure_cmd_dtptb(subparser):
    # create the parser for the 'graphify' command
    parser_ggsolver_solvers = subparser.add_parser("dtptb")
    parser_ggsolver_solvers.add_argument("graph", type=str, default=None,
                                         help="Graph file containing game graph.")
    # parser_ggsolver_solvers.add_argument("-l", "--list", action="store_true",
    #                                      help="List available solvers.")
    parser_ggsolver_solvers.add_argument("-s", "--solver", type=str, default="ggsolver",
                                         help="What solver to use: {'swinreach'}")
    parser_ggsolver_solvers.add_argument("-b", "--backend", type=str, default="ggsolver",
                                         help="Which backend to use: {'ggsolver', 'pgsolver'}")
    parser_ggsolver_solvers.add_argument("-p", "--player", type=int, default=1,
                                         help="Player with reachability objective.")
    parser_ggsolver_solvers.add_argument("-d", "--directory", type=str, default="out/",
                                         help="Directory for intermediate file outputs.")
    parser_ggsolver_solvers.add_argument("-f", "--filename", type=str, default=f'dtptb_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}',
                                         help="Filename for intermediate file outputs.")
    parser_ggsolver_solvers.add_argument("-k", "--protocol", type=str, default='json',
                                         help="Protocol for decoding graph.")
    parser_ggsolver_solvers.add_argument("-v", "--verbosity", type=int, default=0,
                                         help="Verbosity level to use for graphify.")


def configure_cmd_dotviz(subparser):
    # create the parser for the 'dotviz' command
    parser_ggsolver_solvers = subparser.add_parser("dotviz")
    # Arguments
    parser_ggsolver_solvers.add_argument("inp", type=str,
                                         help="Path of Graph input file.")
    parser_ggsolver_solvers.add_argument("out", type=str,
                                         help="Path to DOT file output.")
    parser_ggsolver_solvers.add_argument("-np", nargs="+",
                                         help="Node properties to show in DOT file.")
    parser_ggsolver_solvers.add_argument("-ep", nargs="+",
                                         help="Edge properties to show in DOT file.")
    parser_ggsolver_solvers.add_argument("-k", "--protocol", type=str, default='json',
                                         help="Protocol for decoding graph.")
    parser_ggsolver_solvers.add_argument("-s", "--style", type=str, default='simple',
                                         help="Style to generate dot: {'simple', 'aut', 'solution'}.")


def main():
    # create the top-level parser
    parser = argparse.ArgumentParser()

    # create the subparsers and their respective arguments
    subparsers = parser.add_subparsers(dest="command")

    # configure commands
    configure_cmd_graphify(subparsers)
    configure_cmd_dtptb(subparsers)
    configure_cmd_dotviz(subparsers)

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # process arguments
    if args.command == "graphify":
        cmd_graphify(args)
    elif args.command == "dtptb":
        cmd_dtptb(args)
    elif args.command == "dotviz":
        cmd_dotviz(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == '__main__':
    main()
