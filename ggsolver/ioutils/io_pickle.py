import pickle


def to_pickle(fpath, obj_dict):
    """
    Save the given dictionary to pickle file.

    :param fpath:
    :param obj_dict:
    :return:
    """
    with open(fpath, "wb") as file:
        pickle.dump(obj_dict, file)


def from_pickle(fpath):
    """
    Loads an object dictionary from JSON file.
    :param fpath:
    :return:
    """
    with open(fpath, "rb") as file:
        obj_dict = pickle.load(file)
    return obj_dict


