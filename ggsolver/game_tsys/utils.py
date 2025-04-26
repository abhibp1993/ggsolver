from ggsolver.game_tsys.constants import ModelType


def is_deterministic(model_type: ModelType):
    if model_type in [ModelType.DTPTB]:
        return True
    return False


def is_stochastic(model_type: ModelType):
    if model_type in [ModelType.MDP]:
        return True
    return False


def is_concurrent(model_type: ModelType):
    if model_type in [ModelType.CSG]:
        return True
    return False


def is_turn_based(model_type: ModelType):
    if model_type in [ModelType.DTPTB]:
        return True
    return False
