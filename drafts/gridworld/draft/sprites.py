import sys

import pygame

class BGSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, pos_x, pos_y, color):
        super(BGSprite, self).__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = [pos_x, pos_y]
        pygame.draw.rect(
            self.image,
            (0, 255, 0),
            pygame.Rect(0, 0, self.rect.width, self.rect.height),
            2
        )


class ImageSprite(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(ImageSprite, self).__init__()
        image = pygame.image.load("0.png").convert_alpha()
        self.image = pygame.Surface([50, 50])
        self.rect = self.image.get_rect()
        pygame.transform.scale(image, (50, 50), self.image)
        self.rect.center = (50, 50)

pygame.init()
clock = pygame.time.Clock()

screen_width = 200
screen_height = 200
screen = pygame.display.set_mode([screen_width, screen_height])

sprite = BGSprite(50, 50, 50, 50, (255, 255, 255))
img_sprite = ImageSprite(10, 10)

group = pygame.sprite.Group()
group.add(sprite)
group.add(img_sprite)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.flip()
    group.draw(screen)
    clock.tick()