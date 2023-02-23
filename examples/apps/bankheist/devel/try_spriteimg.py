import pygame as pg
import os
import pathlib
import pygame


pg.init()
# The screen/display has to be initialized before you can load an image.
screen = pg.display.set_mode((640, 480))

curr_file_path = pathlib.Path(__file__).parent.parent.resolve()
img = os.path.join(curr_file_path, "sprites", "cars", "redcar", "car01_0000.png")
IMAGE = pg.image.load(img).convert_alpha()


class Player(pg.sprite.Sprite):

    def __init__(self, pos):
        super().__init__()
        self.image = IMAGE
        self.rect = self.image.get_rect(center=pos)


p = Player((0, 0))
pg.init()
clock = pg.time.Clock()
running = True
window = pg.display.set_mode((640, 480))
window.fill((255, 255, 255))
btn = pg.Rect(0, 0, 100, 30)
rect1 = pg.Rect(0, 30, 100, 100)

# Variable to keep the main loop running
running = True

# Main loop
while running:
    # Look at every event in the queue
    for event in pygame.event.get():
        # Did the user hit a key?
        if event.type == pygame.KEYDOWN:
            # Was it the Escape key? If so, stop the loop.
            if event.key == pygame.K_ESCAPE:
                running = False

        # Did the user click the window close button? If so, stop the loop.
        elif event.type == pygame.QUIT:
            running = False

        pygame.draw.rect(window, (255, 0, 255), rect1, 1)
        window.blit(p.image, (10, 10))
        pygame.display.flip()
