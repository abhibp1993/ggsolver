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
    [0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0],
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
            "N": "sprites/jerry/jerry_face_sprite.png",
            "E": "sprites/jerry/jerry_face_sprite.png",
            "S": "sprites/jerry/jerry_face_sprite.png",
            "W": "sprites/jerry/jerry_face_sprite.png",
            "NE": "sprites/jerry/jerry_face_sprite.png",
            "SE": "sprites/jerry/jerry_face_sprite.png",
            "NW": "sprites/jerry/jerry_face_sprite.png",
            "SW": "sprites/jerry/jerry_face_sprite.png",
        },
    },
    "tom": {
        "sprites": {
            # Direction of facing
            "N": "sprites/tom/tom_sprite.png",
            "E": "sprites/tom/tom_sprite.png",
            "S": "sprites/tom/tom_sprite.png",
            "W": "sprites/tom/tom_sprite.png",
            "NE": "sprites/tom/tom_sprite.png",
            "SE": "sprites/tom/tom_sprite.png",
            "NW": "sprites/tom/tom_sprite.png",
            "SW": "sprites/tom/tom_sprite.png",
        },
    },
    "cheese": {
        "sprites": {
            "front": "sprites/cheese/cheese.png",
        },
        "cheese.1": [8, 4],
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

