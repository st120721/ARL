import gym
import os
import numpy as np
import pyglet
from gym.utils import seeding
from pyglet.gl import *


class Drone2D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, battery_capacity=50, map_path=os.path.dirname(__file__) + "/resources/drone_2d/map.txt"):
        # load map
        try:
            f = open(map_path)
        except IOError:
            print("No map file.")
        else:
            with f:
                map = [l.replace("\n", "").split(" ") for l in f.readlines() if l[0] != "#"]
                map.reverse()
                self.map = np.array(map)
        self.map_size = self.map.shape
        self.battery_capacity = battery_capacity

        self.start_position = np.argwhere(self.map == "S")[0]
        self.goal_position = np.argwhere(self.map == "D")[0]
        self.pick_up_stations = np.argwhere(self.map == "P")
        self.p1 = 0
        self.p2 = 0.1

        self.seed()
        self.viewer = None
        self.state = None
        self.new_package_position = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = {"position": self.start_position,
                      "is_delivered": False,
                      "is_capacity": True,
                      "is_picked": False,
                      "battery_level": self.battery_capacity}
        self.track = []
        return self.state

    def get_allowed_actions(self, state):
        """
        get a list of allowed actions for a state
        action: idle(0), up(1), down(2), left(3), right(4)

        Args:
            state (dict): input state

        Returns:
            actions (list): a list of allowed actions for the input state

        """
        if self.isDone(state):  # being idle in terminal state
            return [0]
        else:
            around_states = lambda i, j: {1: (i + 1, j), 2: (i - 1, j), 3: (i, j - 1), 4: (i, j + 1)}
            actions = [k for k, v in around_states(*state["position"]).items()
                       if (0 <= v[0] < self.map_size[0] and 0 <= v[1] < self.map_size[1]) and self.map[v] != "1"]
            return actions

    def isDone(self, state):
        if state["battery_level"] < 0:
            return True
        elif np.all(state["position"] == self.start_position) and state["is_delivered"]:
            return True

    def getReward(self, state):
        return 0

    def step(self, action):
        info = {}
        if self.isDone(self.state) or action == 0:
            return self.state, self.getReward(self.state), True, info

        around_states = lambda i, j: {1: (i + 1, j), 2: (i - 1, j), 3: (i, j - 1), 4: (i, j + 1)}
        self.state["position"] = around_states(*self.state["position"])[action]
        self.state["battery_level"] -= 1
        # wind
        wind_direction, wind_probability = None, self.np_random.random()
        if wind_probability < self.p1:
            wind_direction = self.np_random.randint(1, 5)
            wind_state = around_states(*self.state["position"])[wind_direction]
            if (wind_direction == 1 and action == 2) or (wind_direction == 2 and action == 1) \
                    or (wind_direction == 3 and action == 4) or (wind_direction == 4 and action == 3):
                self.state["battery_level"] -= 1
            elif 0 <= wind_state[0] < self.map_size[0] and 0 <= wind_state[1] < self.map_size[1] and self.map[
                wind_state] != "1":
                self.state["position"] = wind_state

        # new package
        if not self.new_package_position and not self.state["is_picked"] :
            self.new_package_position = self.pick_up_stations[self.np_random.choice(3)]
            self.new_package_position = (7, 9)

        if np.all(self.state["position"] == self.goal_position):
            self.state["is_delivered"] = True
            self.state["is_capacity"] = False
        if np.all(self.state["position"] == self.new_package_position) and not self.state["is_capacity"]:
            self.state["is_capacity"] = True
            self.state["is_picked"] = True
            self.new_package_position = None

        self.track.append(self.state)
        info["current state"] = self.state
        info["wind"] = wind_direction
        info["new package position"] = self.new_package_position
        info["track"] = self.track
        return self.state, self.getReward(self.state), self.isDone(self.state), info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = self.Viewer(self)
        self.viewer.render()

    class Viewer(pyglet.window.Window):
        def __init__(self, env):
            super(Drone2D.Viewer, self).__init__(resizable=False, caption='Drone Environment', vsync=False)
            # scale
            self.env = env
            if self.env.map_size[0] > self.env.map_size[1]:
                screen_width = 600
                screen_height = int(screen_width * self.env.map_size[0] / self.env.map_size[1])
            else:
                screen_height = 600
                screen_width = int(screen_height * self.env.map_size[1] / self.env.map_size[0])
            self.set_size(int(screen_width * 1.2), screen_height)

            self.grid_size = int(screen_width / self.env.map_size[1])
            self.tau = 100

            glClearColor(1, 1, 1, 1)
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            self.board_batch = pyglet.graphics.Batch()
            self.grid_batch = pyglet.graphics.Batch()
            self.trees_batch = pyglet.graphics.Batch()
            board_points = [(0.01 * screen_width, 0.01 * screen_height),
                            (0.99 * screen_width, 0.01 * screen_height),
                            (0.99 * screen_width, 0.99 * screen_height),
                            (0.01 * screen_width, 0.99 * screen_height),
                            (0.01 * screen_width, 0.01 * screen_height)]
            for i in range(4):
                self.board_batch.add(2, pyglet.gl.GL_LINES, None, ('v2f', (*board_points[i], *board_points[i + 1])),
                                     ('c3B', (0, 0, 0) * 2))
            self.grid_x = np.linspace(0.01 * screen_width, 0.99 * screen_width, self.env.map_size[1], endpoint=False)
            self.grid_y = np.linspace(0.01 * screen_height, 0.99 * screen_height, self.env.map_size[0], endpoint=False)
            for i in np.delete(self.grid_x, 0):
                self.grid_batch.add(2, pyglet.gl.GL_LINES, None,
                                    ('v2f', (i, 0.01 * screen_height, i, 0.99 * screen_height,)),
                                    ('c3B', (119, 136, 153) * 2))
            for j in np.delete(self.grid_y, 0):
                self.grid_batch.add(2, pyglet.gl.GL_LINES, None,
                                    ('v2f', (0.01 * screen_width, j, 0.99 * screen_width, j)),
                                    ('c3B', (119, 136, 153) * 2))
            icon_path = os.path.dirname(__file__)
            house_image1 = pyglet.image.load(icon_path + "/resources/drone_2d/house_1.png")
            self.house1 = pyglet.sprite.Sprite(house_image1, self.grid_x[self.env.goal_position[1]],
                                               self.grid_y[env.goal_position[0]])
            self.house1.update(scale=self.grid_size / house_image1.height)
            house_image2 = pyglet.image.load(icon_path + "/resources/drone_2d/house_2.png")
            self.house2 = pyglet.sprite.Sprite(house_image2, self.grid_x[self.env.goal_position[1]],
                                               self.grid_y[env.goal_position[0]])
            self.house2.update(scale=self.grid_size / house_image2.height)
            tree_image = pyglet.image.load(icon_path + "/resources/drone_2d/tree.png")
            self.trees = []
            for p in np.argwhere(self.env.map == "1"):
                tree = pyglet.sprite.Sprite(tree_image, self.grid_x[p[1]],
                                            self.grid_y[p[0]], batch=self.trees_batch)
                tree.update(scale=self.grid_size / tree_image.height)
                self.trees.append(tree)

            package_image = pyglet.image.load(icon_path + "/resources/drone_2d/package.png")
            self.package = pyglet.sprite.Sprite(package_image)
            self.package.update(scale=self.grid_size / package_image.height)

            self.drone1 = pyglet.sprite.Sprite(package_image)
            drone_image = pyglet.image.load(icon_path + "/resources/drone_2d/drone.png")
            self.drone1 = pyglet.sprite.Sprite(drone_image, self.grid_x[self.env.start_position[1]],
                                               self.grid_y[env.start_position[0]])
            self.drone1.update(scale=self.grid_size / drone_image.height)
            drone_image2 = pyglet.image.load(icon_path + "/resources/drone_2d/drone_2.png")
            self.drone2 = pyglet.sprite.Sprite(drone_image2)
            self.drone2.update(scale=self.grid_size / drone_image2.height)
            self.isReached = False

        def render(self):
            while not self.isReached:
                self.update_map()
                self.switch_to()
                self.dispatch_events()
                self.dispatch_event('on_draw')
                self.flip()

            self.isReached = False
            self.house = self.house2 if self.env.state["is_delivered"] else self.house1

        def on_draw(self):
            self.clear()
            glLineWidth(2)
            self.board_batch.draw()
            glEnable(GL_LINE_STIPPLE)
            glLineWidth(1)
            glLineStipple(1, 0x00ff)
            self.grid_batch.draw()
            glDisable(GL_LINE_STIPPLE)
            self.trees_batch.draw()
            if self.env.state["is_delivered"]:
                self.house2.draw()
            else:
                self.house1.draw()
            if self.env.new_package_position is not None:
                # if not (np.any(self.env.state["position"] == self.env.new_package_position) and not self.env.state["is_capacity"]):
                self.package.x = self.grid_x[self.env.new_package_position[1]]
                self.package.y = self.grid_y[self.env.new_package_position[0]]
                self.package.draw()
            if self.env.state["is_capacity"]:
                self.drone1.draw()
            else:
                self.drone2.draw()

        def update_map(self):

            # self.drone = self.drone1 if self.env.state["is_capacity"] else self.drone2
            new_position = (self.grid_x[self.env.state["position"][1]], self.grid_y[self.env.state["position"][0]])
            dt = pyglet.clock.tick() * self.tau
            if not np.all(np.around(new_position, decimals=0) == np.around((self.drone1.x, self.drone1.y), decimals=0)):
                self.drone1.x += np.sign(new_position[0] - self.drone1.x) * dt
                self.drone2.x = self.drone1.x
                self.drone1.y += np.sign(new_position[1] - self.drone1.y) * dt
                self.drone2.y = self.drone1.y
            else:
                self.isReached = True

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
