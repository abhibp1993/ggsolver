"""
Description: The file provides an interface to define a Tom and Jerry game
    (https://en.wikipedia.org/wiki/Bank_Heist_%28Atari_2600%29). The output is a
    game configuration file (.conf) with following structure.

    ```json
    {
        "game": "Tom and Jerry",
        "metric": "manhattan",
        "terrain": <List of Lists (serialized np.array)>,
        "p1": {
            "sprite": <Relative Path to Sprite Image>,
            "gas capacity": <Number>,
            "init_pos": [x, y],
            "gas": <number>,
            "actions": ["N", "E", "S", "W"],
            "active": <bool>,
        },
        "p2": {
            "sprite": <Relative Path to Sprite Image>,
            "p2.1": {
                "accessible region": <List of Lists (serialized np.array)>,
                "init_pos": [x, y],
                "active": <bool>,
            },
            "p2.2": {
                "accessible region": <List of Lists (serialized np.array)>,
                "init_pos": [x, y],
                "active": <bool>,
            },
        },
        "cheese": {
            "cheese.1": [x, y],
            "cheese.2": [x, y],
        },
        "metadata": {
            "author": <string>,
            "datetime": <date and time of generating this file>,
            "ggsolver version": <ggsolver version>.
        }
    }
    ```

Note: `.conf` files are JSON-compatible.
"""

import numpy as np
import os
import time
import ggsolver
import json


# In this example, I will assume 9x6 world.
#   In cheese heist game, the borders are always walls.
GW_DIM = (11, 8)
METRIC = 'cityblock'        # Metric from scipy.spatial.distance
TERRAIN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]).tolist()

GAME_CONFIG = {
    "game": "Tom and Jerry",
    "metric": "manhattan",
    "terrain": TERRAIN,
    "jerry": {
        "sprites": {
            # Direction of facing
            "N": "sprites/cars/redcar/car01iso_0005.png",
            "E": "sprites/cars/redcar/car01iso_0007.png",
            "S": "sprites/cars/redcar/car01iso_0001.png",
            "W": "sprites/cars/redcar/car01iso_0003.png",
            "NE": "sprites/cars/redcar/car01iso_0006.png",
            "SE": "sprites/cars/redcar/car01iso_0004.png",
            "NW": "sprites/cars/redcar/car01iso_0000.png",
            "SW": "sprites/cars/redcar/car01iso_0002.png",
        },
    },
    "tom": {
        "sprites": {
            # Direction of facing
            "N": "sprites/cars/police/policeiso_0005.png",
            "E": "sprites/cars/police/policeiso_0007.png",
            "S": "sprites/cars/police/policeiso_0001.png",
            "W": "sprites/cars/police/policeiso_0003.png",
            "NE": "sprites/cars/police/policeiso_0006.png",
            "SE": "sprites/cars/police/policeiso_0004.png",
            "NW": "sprites/cars/police/policeiso_0000.png",
            "SW": "sprites/cars/police/policeiso_0002.png",
        },
    },
    "cheese": {
        "sprites": {
            "front": "sprites/cheese/cheese.jpg",
        },
        "cheese.1": [3, 4],
    },
    "metadata": {
        "author": os.getenv('username'),
        "datetime": time.strftime("%Y-%m-%d %H:%M"),
        "ggsolver version": ggsolver.__version__,
    }
}


if __name__ == '__main__':
    with open(f"saved_games/game_{time.strftime('%Y_%m_%d_%H_%M')}.conf", "w") as file:
        json.dump(GAME_CONFIG, file, indent=2)
