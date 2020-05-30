import json

import gym
import os
import pyrr
from gym.utils import seeding
import glfw
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from math import cos, sin


class Drone3D(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, mode="Pathfinder", map=os.path.dirname(__file__) + "/resources/drone_3d/map_1.npy",
                 tracking_path=os.path.dirname(__file__) + "/results/pathfinder.json",
                 map_model=os.path.dirname(__file__) + "/resources/drone_3d/map_7.obj",
                 map_texture=os.path.dirname(__file__) + "/resources/drone_3d/map_1.png"):

        if mode == "Pathfinder":
            try:
                self.map = np.load(map)
                self.map_size = self.map.shape

            except IOError:
                print("No map file.")
        elif mode == "PID_Controller":
            try:
                with open(tracking_path, "r") as f:
                    self.tracking_path = json.load(f)
            except IOError:
                print("No Path file.")
        else:
            raise Exception("Only support Pathfinder and PID_Controller.")

        self.mode = mode
        self.map_model = map_model
        self.map_texture = map_texture
        self.seed()
        self.viewer = None
        self.state = None

        # drone parameter
        self.delta_t = 0.1  # time step
        self.g = 9.8  # acceleration of gravity
        self.m = 1.2  # drone mass
        self.Iy = 0.05  # inertia y
        self.Ix = 0.05  # inertia x
        self.Iz = 0.1  # inertia z
        self.Ct = 10 ^ -4  # integrated thrust coefficient
        self.l = 1  # distance from motor to drone center
        self.Cm = 10 ^ -6  # integrated torque
        self.Jm = 0.01  # total moments of inertia
        self.k1 = 0.02  # drag coefficient x
        self.k2 = 0.02  # drag coefficient y
        self.k3 = 0.02  # drag coefficient z
        self.k4 = 0.1  # drag moment coefficient x
        self.k5 = 0.1  # drag moment coefficient y
        self.k6 = 0.1  # drag moment coefficient z
        self.omegaMax = 330

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.mode == "Pathfinder":
            # start position
            # self.state = dict(position=(0, 0, 1), direction=(0, 1, 0))
            self.state = dict(position=(0, 0, 1))
            self.goal = (self.map_size[0] - 1, self.map_size[1] - 1, 1)
            self.track = [self.state]
        elif self.mode == "PID_Controller":
            self.state = dict(position=(5, 5, 10), angular=(0, 0, 0),
                              velocity=(0, 0, 0), angular_velocity=(0, 0, 0))

    def get_allowed_actions(self, state):
        """
        get a list of allowed actions for a state
        action: idle(0), up(1), down(2), left(3), right(4)

        Args:
            state (dict): input state

        Returns:
            actions (list): a list of allowed actions for the input state

        """
        p = state["position"]
        if state == self.goal:  # being idle in terminal state
            return [0]
        self.around_states = lambda x, y, z: {1: (x + 1, y, z), 2: (x - 1, y, z), 3: (x, y - 1, z), 4: (x, y + 1, z),
                                              5: (x, y, z + 1), 6: (x, y, z - 1),

                                              7: (x + 1, y, z + 1), 8: (x - 1, y, z + 1), 9: (x, y - 1, z + 1),
                                              10: (x, y + 1, z + 1),
                                              11: (x + 1, y, z - 1), 12: (x - 1, y, z - 1), 13: (x, y - 1, z - 1),
                                              14: (x, y + 1, z - 1),

                                              15: (x + 1, y + 1, z), 16: (x + 1, y - 1, z), 17: (x - 1, y + 1, z),
                                              18: (x - 1, y - 1, z),
                                              19: (x + 1, y + 1, z + 1), 20: (x + 1, y - 1, z + 1),
                                              21: (x - 1, y + 1, z + 1),
                                              22: (x - 1, y - 1, z + 1),
                                              23: (x + 1, y + 1, z - 1), 24: (x + 1, y - 1, z - 1),
                                              25: (x - 1, y + 1, z - 1),
                                              26: (x - 1, y - 1, z - 1)}

        actions = []

        for k, v in self.around_states(*p).items():
            if 0 <= v[0] < self.map_size[0] and 0 <= v[1] < self.map_size[1] and 0 <= v[2] < self.map_size[2] and \
                    self.map[v] != 1:
                if k in range(1, 7):
                    actions.append(k)
                elif k in range(7, 15) and np.all(self.map[[p[0], v[0]], [p[1], v[1]], [v[2], p[2]]] != 1):
                    actions.append(k)
                elif k in range(15, 19) and np.all(self.map[[p[0], v[0]], [v[1], p[1]], v[2]] != 1):
                    actions.append(k)
                elif k in range(19, 27) and np.all(self.map[[p[0], v[0]], [v[1], p[1]], v[2]] != 1) and np.all(
                        self.map[[p[0], v[0]], [v[1], p[1]], p[2]] != 1) and np.all(
                    self.map[[p[0], v[0]], [p[1], v[1]], [v[2], p[2]]] != 1):
                    actions.append(k)

        return actions

    def get_reward(self, state):
        return 0

    def kinematics(self, state, omega):
        M = np.array([[self.Ct, self.Ct, self.Ct, self.Ct],
                      [0, -self.l * self.Ct, 0, self.l * self.Ct]
                      [self.l * self.Ct, 0, -self.l * self.Ct, 0]
                      [self.Cm, -self.Cm, self.Cm, -self.Cm]])
        omega_2 = np.square(omega)
        omega_ = -omega(0) + omega(1) - omega(2) + omega(3)
        u = np.dot(M, omega_2)

        x, y, z = state["position"]
        vx, vy, vz = state["velocity"]
        phi, theta, psi = state["angular"]
        vphi, vtheta, vpsi = state["angular_velocity"]

        vx = vx + self.delta_t * (u[0, :] * (cos(psi) * sin(theta) * cos(phi) +
                                             sin(psi) * sin(phi)) / self.m - self.k1 * vx / self.m)
        x = x + self.delta_t * vx

        vy = vy + self.delta_t * (u[0, :] * (sin(psi) * sin(theta) * cos(phi) -
                                             cos(psi) * sin(phi)) / self.m - self.k2 * vy / self.m)
        y = y + self.delta_t * vy

        vz = vz + self.delta_t * (u[0, :] * (cos(theta) * cos(phi)) / self.m - self.g - self.k3 * vz / self.m)
        z = z + self.delta_t * vz

        vphi = vphi + self.delta_t * ((self.Iy - self.Iz) / self.Ix * vtheta * vpsi - self.Jr * vtheta * omega_ +
                                      u[1, :] * self.l / self.Ix - self.k4 * vphi * self.l / self.Ix)
        phi = phi + self.delta_t * vphi

        vtheta = vtheta + self.delta_t * ((self.Iz - self.Ix) / self.Iy * vphi * vpsi - self.Jr * vphi * omega_ + u[2,
                                                                                                                  :] * self.l / self.Iy - self.k5 * vtheta * self.l / self.Iy)
        theta = theta + self.delta_t * vtheta

        vpsi = vpsi + self.delta_t * (
                (self.Ix - self.Iy) / self.Iz * vphi * vtheta + u[3, :] * 1 / self.Iz - self.k6 *
                vpsi / self.Iz)
        psi = psi + self.delta_t * vpsi

        state = dict(position=(x, y, z), angular=(phi, theta, psi), velocity=(vx, vy, vz),
                     angular_velocity=(vphi, vtheta, vpsi))

        return state

    def is_done(self, state):
        if np.all(state["position"] == self.goal):
            return True

    def save_path(self):
        with open(os.path.dirname(__file__) + "/results/pathfinder.json", "wt") as f:
            json.dump(self.track, f)

    def step(self, action):
        if self.mode == "Pathfinder":
            info = {}
            if self.is_done(self.state) or action == 0:
                return self.state, self.get_reward(self.state), True, info

            if action not in self.get_allowed_actions(self.state):
                raise Exception("Action is not allowed.")
            state = {}
            state["position"] = self.around_states(*self.state["position"])[action]

            # state["direction"] = self.around_states(0, 0, 0)[action]
            self.track.append(state)
            self.state = state

        elif self.mode == "PID_Controller":

            pass

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = self.Viewer(self)
        self.viewer.render()

    class Viewer:
        def __init__(self, env):
            self.env = env
            if not glfw.init():
                raise Exception("glfw can not be initialized!")

            self.width = 800
            self.height = 800
            glfwWindowHint(GLFW_SAMPLES, 4)  # Anti-Aliasing
            self.window = glfw.create_window(self.width, self.height, "Drone 3D", None, None)
            glfw.set_window_size_callback(self.window, self.window_resize)
            glfw.set_key_callback(self.window, self.key_input)
            glfw.make_context_current(self.window)

            vertex_src = """
                # version 330
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec2 texture;
                layout(location = 2) in vec3 normal;

                uniform mat4 model;
                uniform mat4 projection;
                uniform mat4 view;

                out vec2 v_texture;
                out vec3 f_normal;
                out vec3 f_position;

                void main()
                {   

                    gl_Position = projection * view *  model * vec4(position, 1.0f);
                    f_position = vec3(model * vec4(position, 1.0f));
                    f_normal = mat3(transpose(inverse(model))) * normal;
                    v_texture = texture;
                }
                """
            fragment_src = """
                # version 330
                in vec2 v_texture;
                in vec3 f_normal;
                in vec3 f_position;

                out vec4 out_color;

                uniform sampler2D s_texture;
                uniform vec3 view_position;
                uniform vec3 light_position;
                uniform vec3 light_color;

                void main()
                {           
                    vec3 color = texture(s_texture, v_texture).rgb;
                    vec3 ambient = 0.33 * color;
                    vec3 light_direction = normalize(light_position - f_position);
                    vec3 normal = normalize(f_normal);
                    float diff = max(dot(light_direction, normal), 0.0);
                    vec3 diffuse = diff * color;
                    vec3 view_direction = normalize(view_position - f_position);
                    vec3 reflectDir = reflect(-light_direction, normal);
                    float spec = 0.0;
                    vec3 halfway = normalize(light_direction + view_direction);  
                    spec = pow(max(dot(normal, halfway), 0.0), 32.0);
                    vec3 specular = vec3(0.3) * spec; 
                    out_color = vec4(ambient + diffuse + specular, 1.0);
                }
                """
            vertex2_src = """
            # version 330 
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 a_color;

                uniform mat4 model;
                uniform mat4 projection;
                uniform mat4 view;

                out vec3 v_color;

                void main()
                {   

                    gl_Position = projection * view *  model * vec4(position, 1.0f);         
                    v_color = a_color;
                }
            """

            fragment2_src = """
            # version 330

            in vec3 v_color;
            out vec4 out_color;

            void main()
            {
                out_color = vec4(v_color, 1.0);
            }
            """
            model_path = os.path.dirname(__file__)
            self.drone_indices, self.drone_buffer = self.load_model(model_path + "/resources/drone_3d/drone.obj")
            self.map_indices, self.map_buffer = self.load_model(self.env.map_model)
            self.textures = glGenTextures(2)
            self.load_texture(model_path + "/resources/drone_3d/drone.png", self.textures[0])
            self.load_texture(self.env.map_texture, self.textures[1])

            self.VAO = glGenVertexArrays(2)
            self.VBO = glGenBuffers(2)

            # drone VAO, VBO
            glBindVertexArray(self.VAO[0])
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO[0])
            glBufferData(GL_ARRAY_BUFFER, self.drone_buffer.nbytes, self.drone_buffer, GL_STATIC_DRAW)

            # drone vertices, textures, normals
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.drone_buffer.itemsize * 8, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.drone_buffer.itemsize * 8, ctypes.c_void_p(12))
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.drone_buffer.itemsize * 8, ctypes.c_void_p(20))

            # map VAO, VBO
            glBindVertexArray(self.VAO[1])
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO[1])
            glBufferData(GL_ARRAY_BUFFER, self.map_buffer.nbytes, self.map_buffer, GL_STATIC_DRAW)

            # map vertices, textures, normals
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.map_buffer.itemsize * 8, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.map_buffer.itemsize * 8, ctypes.c_void_p(12))
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.map_buffer.itemsize * 8, ctypes.c_void_p(20))
            glEnableVertexAttribArray(2)

            shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                    compileShader(fragment_src, GL_FRAGMENT_SHADER))
            glUseProgram(shader)
            glClearColor(135 / 255, 206 / 255, 235 / 255, 1)  # sky color
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.model_location = glGetUniformLocation(shader, "model")
            self.projection_location = glGetUniformLocation(shader, "projection")
            self.view_location = glGetUniformLocation(shader, "view")
            self.view_position = glGetUniformLocation(shader, "view_position")
            self.light_position = glGetUniformLocation(shader, "light_position")
            self.light_color = glGetUniformLocation(shader, "light_color")
            self.map_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]), dtype=float)

            ####
            vertices = [-10, -10, 0.0, 1.0, 0.0, 0.0,
                        10, -10, 0.0, 0.0, 1.0, 0.0,
                        -10, 10, 0.0, 0.0, 0.0, 1.0,
                        10, 10, 0.0, 1.0, 1.0, 1.0,
                        0.0, 20, 0.0, 1.0, 1.0, 0.0]

            # indices = [0, 1, 2,
            #            1, 2, 3,
            #            2, 3, 4]
            indices = [0, 1, 1, 2, 2, 3, 3, 4]
            vertices = np.array(vertices, dtype=np.float32)
            self.indices = np.array(indices, dtype=np.uint32)

            P_VBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, P_VBO)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            # Element Buffer Object
            # EBO = glGenBuffers(1)
            # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
            # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

            # glEnableVertexAttribArray(0)
            # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

            # glEnableVertexAttribArray(1)
            # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

            glUniform3f(self.light_position, -200, 200, 200)
            glUniform3f(self.light_color, 1, 1, 1)

            self.follow_mode = True

        def render(self):
            if self.env.mode == "Pathfinder":
                # drone position
                # position, front = self.trans_coord(self.env.track[-2])
                # new_position, new_front = self.trans_coord(self.env.track[-1])
                position = self.trans_coord(self.env.track[-2])
                new_position = self.trans_coord(self.env.track[-1])
                tau = 0.008

                while 1:
                    # if (not np.all(np.around(front, decimals=2) == np.around(new_front, decimals=2))) \
                    #         and (not np.all(np.zeros(3)==new_front)):
                    #     rot = pyrr.Matrix33.from_y_rotation(-0.001)
                    #     f = pyrr.vector3.normalise(pyrr.matrix33.multiply(rot, front))
                    #     # p = position
                    #     front=f
                    # else:
                    #     f=front if np.all(np.zeros(3)==new_front) else new_front
                    #     position= position + (t)*0.001

                    position = position + np.sign(new_position - position) * tau
                    self.update(position, [-1, 0, -1])

                    glfw.poll_events()
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glUniformMatrix4fv(self.view_location, 1, GL_FALSE, self.view)
                    glUniform3f(self.view_position, self.camera_position.x, self.camera_position.y,
                                self.camera_position.z)
                    glUniformMatrix4fv(self.projection_location, 1, GL_FALSE, self.projection)

                    glBindVertexArray(self.VAO[0])
                    glBindTexture(GL_TEXTURE_2D, self.textures[0])

                    glUniformMatrix4fv(self.model_location, 1, GL_FALSE, self.drone_position)
                    glDrawArrays(GL_TRIANGLES, 0, len(self.drone_indices))

                    glBindVertexArray(self.VAO[1])
                    glBindTexture(GL_TEXTURE_2D, self.textures[1])
                    glUniformMatrix4fv(self.model_location, 1, GL_FALSE, self.map_position)
                    glDrawArrays(GL_TRIANGLES, 0, len(self.map_indices))
                    glfw.swap_buffers(self.window)

                    # glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_INT, None)

                    if np.all(np.around(position, decimals=2) == np.around(new_position, decimals=2)):
                        break

        def update(self, position, front):
            self.drone_position = pyrr.matrix44.create_from_translation(position)
            scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([0.7, 0.7, 0.7]))
            self.drone_position = pyrr.matrix44.multiply(scale, self.drone_position)

            if self.follow_mode:
                self.projection = pyrr.matrix44.create_perspective_projection_matrix(75,
                                                                                     self.width / self.height,
                                                                                     0.1, 1000)

                front = pyrr.vector.normalise(front)
                self.camera_position = pyrr.Vector3(position - front * 15)

                self.camera_position.y += 10
                self.target_position = position + pyrr.Vector3([0, 6, 0])
                self.view = pyrr.matrix44.create_look_at(self.camera_position, self.target_position,
                                                         pyrr.Vector3([0, 1, 0]))
            else:
                # scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([6, 6, 6]))
                # self.map_position = pyrr.matrix44.create_from_translation(
                #     pyrr.Vector3([self.height / 2, 0, self.width / 2]),dtype=float)
                # self.map_position = pyrr.matrix44.multiply(scale, self.map_position)
                # self.drone_position = pyrr.matrix44.create_from_translation(
                #     position) + pyrr.matrix44.create_from_translation(
                #     pyrr.Vector3([self.height / 2, 0, self.width / 2]),dtype=float)
                # self.drone_position = pyrr.matrix44.multiply(scale, self.drone_position)
                # self.projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, self.width, 0,
                #                                                                      self.height, -1000, 1000)
                self.projection = pyrr.matrix44.create_perspective_projection_matrix(20,
                                                                                     self.width / self.height,
                                                                                     30, 1000)

                self.camera_position = pyrr.Vector3([0, 650, 0])
                self.target_position = pyrr.Vector3([0, 0, 0])
                self.view = pyrr.matrix44.create_look_at(self.camera_position, self.target_position,
                                                         pyrr.Vector3([-1, 0, 0]))

        def trans_coord(self, state):

            if self.env.mode == "Pathfinder":
                p = state["position"]
                # d = state["direction"]
                position = np.array(
                    [p[1] - self.env.map_size[1] / 2 + 0.5,
                     -p[2], p[0] - self.env.map_size[0] / 2 + 0.5]) * -10
                # front = np.array([-d[1], d[2], -d[0]])

            return position
            #     front = np.array([-d[1], 0, -d[0]])
            # return position, front

        def window_resize(self, window, width, height):
            glViewport(0, 0, width, height)
            self.width = width
            self.height = height
            # self.projection = pyrr.matrix44.create_perspective_projection_matrix(75,
            #                                                                     width / height,
            #                                                                      0.1, 1000)
            # glUniformMatrix4fv(self.projection_location, 1, GL_FALSE, self.projection)

        def key_input(self, window, key, scancode, action, mode):
            if key == glfw.KEY_SPACE and action == glfw.PRESS:
                self.follow_mode = not self.follow_mode

        @staticmethod
        def load_model(model_path):
            model = dict(v=[], vt=[], vn=[])
            a_indices, indices = [], []

            with open(model_path, 'r') as f:
                line = f.readline()
                while line:
                    values = line.split()
                    type = values[0]
                    if type == 'f':
                        for value in values[1:]:
                            temp = value.split('/')
                            a_indices.extend(temp)
                            indices.append(int(temp[0]) - 1)
                    elif type in ["v", "vt", "vn"]:
                        model[type].extend(values[1:])

                    line = f.readline()
            data = []
            for i, j in enumerate(a_indices):
                if i % 3 == 0:
                    data.extend(model["v"][(int(j) - 1) * 3:int(j) * 3])
                elif i % 3 == 1:
                    data.extend(model["vt"][(int(j) - 1) * 2:int(j) * 2])
                elif i % 3 == 2:
                    data.extend(model["vn"][(int(j) - 1) * 3:int(j) * 3])
            return np.array(indices, dtype='uint32'), np.array(data, dtype='float32')

        @staticmethod
        def load_texture(texture_path, texture):
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            image = Image.open(texture_path)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            return texture
