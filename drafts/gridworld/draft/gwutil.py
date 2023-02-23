# GLOBALS: OBSTACLE TYPES
GW_OBS_TYPE_SINK = "sink"
GW_OBS_TYPE_BOUNCY = "bouncy"

# GLOBALS: BOUNDARY TYPES
GW_BOUNDARY_TYPE_BOUNCY = "bouncy"
GW_BOUNDARY_TYPE_ROUND = "round"

# GLOBALS: STANDARD ACTIONS
GW_ACT_N = {"N": lambda r, c: (r + 1, c)}
GW_ACT_E = {"E": lambda r, c: (r, c + 1)}
GW_ACT_S = {"S": lambda r, c: (r - 1, c)}
GW_ACT_W = {"W": lambda r, c: (r, c - 1)}
GW_ACT_NE = {"NE": lambda r, c: (r + 1, c + 1)}
GW_ACT_NW = {"NW": lambda r, c: (r + 1, c - 1)}
GW_ACT_SE = {"SE": lambda r, c: (r - 1, c + 1)}
GW_ACT_SW = {"SW": lambda r, c: (r - 1, c - 1)}
GW_ACT_STAY = {"STAY": lambda r, c: (r, c)}

GW_ACT_4 = GW_ACT_N | GW_ACT_E | GW_ACT_S | GW_ACT_W
GW_ACT_5 = GW_ACT_4 | GW_ACT_STAY
GW_ACT_8 = GW_ACT_4 | GW_ACT_NE | GW_ACT_NW | GW_ACT_SE | GW_ACT_SW
GW_ACT_9 = GW_ACT_8 | GW_ACT_STAY


def bouncy_boundary(row, col, dim):
    return max(min(row, dim[0] - 1), 0), max(min(col, dim[1] - 1), 0)


def bouncy_obstacle(row, col, n_row, n_col, obs):
    for o_row, o_col in obs:
        if n_row == o_row and n_col == o_col:
            return row, col
    return n_row, n_col


def is_cell_in_gridworld(row, col, dim):
    return 0 <= row < dim[0] and 0 <= col < dim[1]
