from loguru import logger

try:
    import spot


    def spot_eval(cond, true_atoms):
        """
        Evaluates a propositional logic formula given the set of true atoms.

        :param true_atoms: (Iterable[str]) A propositional logic formula.
        :return: (bool) True if formula is true, otherwise False.
        """

        # Define a transform to apply to AST of spot.formula.
        def transform(node: spot.formula):
            if node.is_literal():
                if "!" not in node.to_str():
                    if node.to_str() in true_atoms:
                        return spot.formula.tt()
                    else:
                        return spot.formula.ff()

            return node.map(transform)

        # Apply the transform and return the result.
        # Since every literal is replaced by true or false,
        #   the transformed formula is guaranteed to be either true or false.
        return True if transform(spot.formula(cond)).is_tt() else False


    logger.success("Spot detected...")

except ImportError:
    logger.critical("Spot is not installed. Automata related functions will not work...")


    def spot_eval(cond, true_atoms):
        raise ImportError("Spot is not installed. Automata related functions will not work...")
