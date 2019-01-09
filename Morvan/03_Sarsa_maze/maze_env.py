"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.
"""
import tkinter as tk
import numpy as np
import time
import sys


UNIT = 40  # pixels
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk):
    def __init__(self):
        super().__init__()
        self.action_space = ["u", "d", "l", "r"]
        self.n_actions = len(self.action_space)
        self.title("maze")
        self.height = MAZE_H * UNIT
        self.width = MAZE_W * UNIT
        self.geometry("{0}x{1}".format(self.height, self.width))
        self.hell1_coord = (2, 1)
        self.hell2_coord = (1, 2)
        self.hell_coord_list = [(2, 1), (1, 2)]
        self.oval_coord = (2, 2)
        self.item_size = 30
        self._build_maze()

    def _get_center_from_origin(self, origin, coord):
        """
        给定 origin坐标和相对坐标，计算目标点的坐标"""
        assert len(origin) == len(coord), "origin's coord num doesn't match coord's"
        x, y = coord
        return origin + np.array([x * UNIT, y * UNIT])

    def _create_item(self, center_coord, size, color, item="rect"):
        """
        给定目标点坐标和图形信息，制造图形
        """
        if item.lower() == "rect":
            obj = self.canvas.create_rectangle(
                int(center_coord[0] - size / 2), int(center_coord[1] - size / 2),
                int(center_coord[0] + size / 2), int(center_coord[1] + size / 2),
                fill=color
            )
        elif item.lower() == "oval":
            obj = self.canvas.create_oval(
                int(center_coord[0] - size / 2), int(center_coord[1] - size / 2),
                int(center_coord[0] + size / 2), int(center_coord[1] + size / 2),
                fill=color
            )
        return obj

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg="white", height=self.height, width=self.width)

        # create grids
        for c in range(0, self.width, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.height
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, self.height, UNIT):
            x0, y0, x1, y1 = 0, r, self.width, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([int(UNIT / 2), int(UNIT / 2)])

        # add hell
        self.hell_list = []
        for hell_coord in self.hell_coord_list:
            hell_center = self._get_center_from_origin(origin, hell_coord)
            hell = self._create_item(hell_center, self.item_size, "black", "rect")
            self.hell_list.append(hell)


        # add oval
        oval_center = self._get_center_from_origin(origin, self.oval_coord)
        self.oval = self._create_item(oval_center, self.item_size, "yellow", "oval")

        # create red rect
        self.rect = self._create_item(origin, self.item_size, "red", "rect")

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([int(UNIT / 2), int(UNIT / 2)])
        self.rect = self._create_item(origin, self.item_size, "red", "rect")

        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == "u":  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == "d":  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == "l":  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == "r":  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next_state

        if s_ == self.canvas.coords(self.oval):  # bingo
            reward = 1
            done = True
            s_ = "terminal"
        elif s_ in [self.canvas.coords(hell) for hell in self.hell_list]:  # dead
            reward = -1
            done = True
            s_ = "terminal"
        elif s_ == s:  # hit the wall
            reward = -0.5
            done = False
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

if __name__ == '__main__':
    env = Maze()
    print(env.canvas.coords(env.oval))
    def update():
        for t in range(10):
            env.reset()
            done = False
            while not done:
                env.render()
                a = np.random.choice(["u", "d", "l", "r"])
                s, r, done = env.step(a)
                time.sleep(0.1)
    env.after(100, update)
    env.mainloop()
    env.destroy()



