import json

import numpy as np
import ggsolver.mdp as mdp
import ggsolver.gridworld as gw
from collections import namedtuple


class BankHeistWindow(gw.Window):
    def __init__(self, name, size, game_config, **kwargs):
        super(BankHeistWindow, self).__init__(name, size, **kwargs)
        with open(game_config, "r") as file:
            self._game_config = json.load(file)

        if self._game_config["game"] != "Bank Heist":
            raise ValueError("The game is not Bank Heist.")

        # Construct grid
        self._terrain = np.array(self._game_config["terrain"])
        grid_size = self._terrain.shape
        self.grid = gw.Grid(
            name="grid1",
            parent=self,
            position=(0, 0),
            size=size,
            grid_size=grid_size,
            backcolor=gw.COLOR_BEIGE,
            anchor=gw.AnchorStyle.TOP_LEFT,
        )

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if self._terrain[x, y] == 0:
                    self.grid[x, y].backcolor = gw.COLOR_GRAY51

    def sm_update(self, event_args):
        print(f"sm_update: {event_args}")


if __name__ == '__main__':
    # conf = f"saved_games/game_2022_11_21_20_05.conf"
    conf = f"E:/Github-Repositories/ggsolver/examples/apps/bankheist/saved_games/game_2022_11_21_20_05.conf"
    window = BankHeistWindow(name="Bank Heist", size=(600, 600), game_config=conf)
    window.run()
