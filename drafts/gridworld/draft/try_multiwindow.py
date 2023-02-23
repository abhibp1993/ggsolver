# Player class with window.show()
# GWSim class showing two windows
# GWSim sending an update message with some parameters.
# GWSim receiving updated state.

import pygame
import sys
from multiprocessing import Lock, Pipe, Process


class GWSim:
    def __init__(self):
        pass


class Player:
    def __init__(self, name, bgcolor):
        self.size = (200, 200)
        self.screen = None
        self.bgcolor = bgcolor
        self.name = name

    def run(self):
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.bgcolor)
        while True:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        sys.exit(0)  # close this specific process

            pygame.display.update()


def main():
    player1 = Player("player 1", (255, 255, 255))
    player2 = Player("player 2", (255, 0, 0))

    process1 = Process(target=player1.run)
    process2 = Process(target=player2.run)

    process1.daemon = True
    process2.daemon = True

    process1.start()
    process2.start()

    while True:
        pass


if __name__ == '__main__':
    # player = Player("player")
    # player.run()
    main()