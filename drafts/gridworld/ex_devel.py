from ggsolver.gridworld.models import *


class SubGrid(Grid):
    def update(self):
        super(SubGrid, self).update()
        # print(self.name, "update")

    def on_key_down(self, event_args):
        # print(f"Call: {self.name}.on_key_down")
        if event_args.key == pygame.K_RIGHT:
            self.move_right_by(5)
        if event_args.key == pygame.K_LEFT:
            self.move_left_by(5)
        if event_args.key == pygame.K_UP:
            self.move_up_by(5)
        if event_args.key == pygame.K_DOWN:
            self.move_down_by(5)

    def on_mouse_click(self, event_args):
        print(f"Call: {self.name}.on_mouse_click")


if __name__ == '__main__':
    window = Window(name="window1", size=(600, 600), backcolor=(245, 245, 220), fps=60)
    # control = Control(name="control1", parent=window, position=(100, 100), size=(50, 10))

    grid = Grid(name="grid", parent=window, position=(0, 0), size=(600, 600), grid_size=(2, 2))
    window.add_control(grid)

    sub_grid = SubGrid(name="sub-grid", parent=grid[0, 0], position=(0, 0), size=(50, 50), grid_size=(2, 1), backcolor=(152, 245, 255))
    grid[0, 0].add_control(sub_grid)

    sub_grid2 = SubGrid(name="sub-grid2", parent=grid[1, 1], position=(0, 0), size=(50, 50), grid_size=(1, 2), backcolor=(188,238,104))
    grid[1, 1].add_control(sub_grid2)

    window.run()
