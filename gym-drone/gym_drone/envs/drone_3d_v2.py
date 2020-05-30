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
                 map_model=os.path.dirname(__file__) + "/resources/drone_3d/map_1.obj",
                 map_texture=os.path.dirname(__file__) + "/resources/drone_3d/map_1.png"):

        if mode == "Pathfinder":
            try:
                self.map = np.load(map)
                self.map_size = self.map.shape
                self.tracking_path = None
            except IOError:
                print("No map file.")
        elif mode == "PID_Controller":
            try:

                ####
                self.map = np.load(map)
                self.map_size = self.map.shape

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
        self.viewer = self.Viewer(self)
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
            self.track = []
        elif self.mode == "PID_Controller":
            self.state = dict(position=(5, 5, 10), angular=(0, 0, 0),
                              velocity=(0, 0, 0), angular_velocity=(0, 0, 0))
            self.track = []

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
        file = "pathfinder.json" if self.mode == "Pathfinder" else "pid_controller.json"
        with open(os.path.dirname(__file__) + "/results/" + file, "wt") as f:
            json.dump(self.track, f)

    test_tracking = np.load(os.path.dirname(__file__) + "/results/test_tracking3.npy")
    loop = 0

    def step(self, action):
        if self.mode == "Pathfinder":
            info = {}
            if self.is_done(self.state) or action == 0:
                return self.state, self.get_reward(self.state), True, info

            if action not in self.get_allowed_actions(self.state):
                raise Exception("Action is not allowed.")
            state = {}
            state["position"] = self.around_states(*self.state["position"])[action]

            self.track.append(state)
            self.state = state

        elif self.mode == "PID_Controller":

            state = {}
            if Drone3D.loop < Drone3D.test_tracking.shape[0]:
                state["position"] = Drone3D.test_tracking[Drone3D.loop]
                Drone3D.loop += 1
                self.track.append(state)
                self.state = state

    def render(self, mode='human'):
        # if self.viewer is None:
        #     self.viewer = self.Viewer(self)
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
                uniform bool is_line;

                out vec2 v_texture;
                out vec3 f_normal;
                out vec3 f_position;
                out vec3 p_color;

                void main()
                {   

                    if(is_line)
                    {
                    gl_Position = projection * view *  model * vec4(position, 1.0f);
                    f_position = vec3(model * vec4(position, 1.0f));
                    f_normal = mat3(transpose(inverse(model))) * normal;
                    v_texture = texture;
                    }
                    else
                    {
                    gl_Position = projection * view * vec4(position, 1.0f);         

                    }

                }
                """
            fragment_src = """
                # version 330
                in vec2 v_texture;
                in vec3 f_normal;
                in vec3 f_position;


                out vec4 out_color;

                uniform bool is_line;
                uniform sampler2D s_texture;
                uniform vec3 camera_position;
                uniform vec3 light_position;
                uniform vec3 light_color;
                uniform vec3 path_color;

                void main()
                {   
                    if(is_line)
                    {      
                    vec3 color = texture(s_texture, v_texture).rgb;
                    vec3 ambient = 0.33 * color;
                    vec3 light_direction = normalize(light_position - f_position);
                    vec3 normal = normalize(f_normal);
                    float diff = max(dot(light_direction, normal), 0.0);
                    vec3 diffuse = diff * color;
                    vec3 view_direction = normalize(camera_position - f_position);
                    vec3 reflectDir = reflect(-light_direction, normal);
                    float spec = 0.0;
                    vec3 halfway = normalize(light_direction + view_direction);  
                    spec = pow(max(dot(normal, halfway), 0.0), 32.0);
                    vec3 specular = vec3(0.3) * spec; 
                    out_color = vec4(ambient + diffuse + specular, 1.0);
                    }
                    else
                    {
                    out_color = vec4(path_color, 1.0);
                    }
                }

                """

            model_path = os.path.dirname(__file__)
            self.drone_indices, self.drone_buffer = self.load_model(model_path + "/resources/drone_3d/drone.obj")
            self.map_indices, self.map_buffer = self.load_model(self.env.map_model)
            self.textures = glGenTextures(2)
            self.load_texture(model_path + "/resources/drone_3d/drone.png", self.textures[0])
            self.load_texture(self.env.map_texture, self.textures[1])

            self.VAO = glGenVertexArrays(4)
            self.VBO = glGenBuffers(4)

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
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, self.map_buffer.itemsize * 8, ctypes.c_void_p(20))

            if self.env.mode == "PID_Controller":
                self.tracking_path = np.ravel([self.trans_coord(p) for p in self.env.tracking_path])
                self.tracking_path = self.tracking_path.astype(np.float32)
                glBindVertexArray(self.VAO[2])
                glBindBuffer(GL_ARRAY_BUFFER, self.VBO[2])
                glBufferData(GL_ARRAY_BUFFER, self.tracking_path.nbytes, self.tracking_path, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

                tracking_path_indices = np.ravel([[i, i + 1] for i in range(int(self.tracking_path.size / 3 - 1))])
                self.tracking_path_indices = np.array(tracking_path_indices, dtype=np.uint32)

                self.T_EBO = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.T_EBO)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.tracking_path_indices.nbytes, self.tracking_path_indices,
                             GL_STATIC_DRAW)

                self.path = []

            shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                    compileShader(fragment_src, GL_FRAGMENT_SHADER))
            glUseProgram(shader)
            glClearColor(135 / 255, 206 / 255, 235 / 255, 1)  # sky color
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.model_location = glGetUniformLocation(shader, "model")
            self.projection = glGetUniformLocation(shader, "projection")
            self.view = glGetUniformLocation(shader, "view")
            self.camera_position = glGetUniformLocation(shader, "camera_position")
            self.light_position = glGetUniformLocation(shader, "light_position")
            self.light_color = glGetUniformLocation(shader, "light_color")
            self.is_line = glGetUniformLocation(shader, "is_line")
            self.path_color = glGetUniformLocation(shader, "path_color")

            self.map_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]), dtype=float)
            glUniform3f(self.light_position, -200, 200, 200)
            glUniform3f(self.light_color, 0.9, 0.9, 0.9)

            self.follow_mode = True

        def render(self):
            if self.env.mode == "Pathfinder":
                position = np.array(self.trans_coord(self.env.track[-2]))
                new_position = np.array(self.trans_coord(self.env.track[-1]))
                tau = 0.01
                while 1:
                    position = position + np.sign(new_position - position) * tau
                    glfw.poll_events()
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                    self.draw(position)

                    glfw.swap_buffers(self.window)
                    if np.all(np.around(position, decimals=2) == np.around(new_position, decimals=2)):
                        break
            elif self.env.mode == "PID_Controller":
                position = np.array(self.trans_coord(self.env.track[-2], False))
                new_position = np.array(self.trans_coord(self.env.track[-1], False))
                tau = 0.01
                while 1:
                    position = position + np.sign(new_position - position) * tau
                    self.path.append(position)
                    glfw.poll_events()

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glUniform1i(self.is_line, False)
                    glEnable(GL_LINE_STIPPLE)
                    glLineWidth(1)
                    # glLineStipple(6, 0xAAAA)
                    glLineStipple(1, 0x00ff)
                    glBindVertexArray(self.VAO[2])
                    glUniform3f(self.path_color, *(1, 0, 0))
                    glDrawElements(GL_LINES, len(self.tracking_path_indices), GL_UNSIGNED_INT, None)

                    path = np.array(self.path, dtype=np.float32).flatten()
                    glBindVertexArray(self.VAO[3])
                    glBindBuffer(GL_ARRAY_BUFFER, self.VBO[3])
                    glBufferData(GL_ARRAY_BUFFER, path.nbytes, path, GL_STATIC_DRAW)
                    glEnableVertexAttribArray(0)
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

                    path_indices = np.ravel([[i, i + 1] for i in range(int(path.size / 3 - 1))])
                    path_indices = np.array(path_indices, dtype=np.uint32)

                    P_EBO = glGenBuffers(1)
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, P_EBO)
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, path_indices.nbytes, path_indices, GL_STATIC_DRAW)
                    glLineWidth(2)
                    glUniform3f(self.path_color, *(0, 0, 1))
                    glDrawElements(GL_LINES, len(path_indices), GL_UNSIGNED_INT, None)

                    self.draw(position)

                    glfw.swap_buffers(self.window)
                    if np.all(np.around(position, decimals=2) == np.around(new_position, decimals=2)):
                        break

        def draw(self, position, front=[1, 1, 0]):
            self.drone_position = pyrr.matrix44.create_from_translation(position)

            if self.follow_mode:
                self.projection_ = pyrr.matrix44.create_perspective_projection_matrix(75,
                                                                                      self.width / self.height,
                                                                                      0.1, 1000)

                front = pyrr.vector.normalise(front)
                # if self.env.mode == "PID_Controller" and len(self.path)>30:
                #     temp=np.array(self.path)[-30:,:]
                #     # position=np.around(np.mean(temp,axis=0), decimals=3)
                #     position=np.mean(temp,axis=0)
                self.camera_ = pyrr.Vector3(position - front * 15)
                self.camera_.z += 10
                self.target_position = position + pyrr.Vector3([0, 0, 6])
                self.view_ = pyrr.matrix44.create_look_at(self.camera_, self.target_position,
                                                          pyrr.Vector3([0, 0, 1]))
            else:
                self.projection_ = pyrr.matrix44.create_perspective_projection_matrix(20,
                                                                                      self.width / self.height,
                                                                                      30, 1000)

                self.camera_ = pyrr.Vector3([0, 0, 650])
                self.target_position = pyrr.Vector3([0, 0, 0])
                self.view_ = pyrr.matrix44.create_look_at(self.camera_, self.target_position,
                                                          pyrr.Vector3([0, 1, 0]))

            glUniformMatrix4fv(self.projection, 1, GL_FALSE, self.projection_)
            glUniformMatrix4fv(self.view, 1, GL_FALSE, self.view_)
            glUniform3f(self.camera_position, self.camera_.x, self.camera_.y,
                        self.camera_.z)
            glUniform1i(self.is_line, True)
            glBindVertexArray(self.VAO[0])
            glBindTexture(GL_TEXTURE_2D, self.textures[0])
            glUniformMatrix4fv(self.model_location, 1, GL_FALSE, self.drone_position)
            glDrawArrays(GL_TRIANGLES, 0, len(self.drone_indices))

            glBindVertexArray(self.VAO[1])
            glBindTexture(GL_TEXTURE_2D, self.textures[1])
            glUniformMatrix4fv(self.model_location, 1, GL_FALSE, self.map_position)
            glDrawArrays(GL_TRIANGLES, 0, len(self.map_indices))

        def trans_coord(self, state, grid_world=True):

            if grid_world:
                p = state["position"]
                position = [(p[0] - self.env.map_size[0] / 2 + 0.5) * 10,
                            (p[1] - self.env.map_size[1] / 2 + 0.5) * 10, p[2] * 10]
            else:
                position = state["position"]
            return position

        def window_resize(self, window, width, height):
            glViewport(0, 0, width, height)
            self.width = width
            self.height = height

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