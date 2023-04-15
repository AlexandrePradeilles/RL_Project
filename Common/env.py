import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SuperMario(gym.Env):
    metadata = {"render_fps": 96}

    def __init__(self, size=5, render_fps=96):
        self.size = size  # The size of the square grid
        self.ratio = self.size*24/(5*6*60)
        self.window_size = 508  # The size of the PyGame window
        self.metadata["render_fps"] = render_fps

        # Observations
        self.observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.window_size, self.window_size, 3),
                    dtype=np.uint8
                )

        # We have 6 actions, corresponding to "right", "up", "left", "nothing", "up+right", "up+left"
        self.action_space = spaces.Discrete(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.vert_speed = 0
        self.horiz_speed = 0
        self.jump_count = 0

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Let's define all the elements of the Super Mario Bros level

        self.pre_obs = [np.array([self.size-1, self.size, -self.size, 0])]

        self.stones = []
        self.ground = []
        self.pipes = []


        self.ground += [np.array([self.size-1, self.size, 0, self.size])]
        self.stones += [np.array([self.size-3, self.size-2.5, 5, 5.5]),
                       np.array([self.size-3, self.size-2.5, 7, 9.5]), np.array([self.size-5, self.size-4.5, 8, 8.5])]
        
        step = self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step])]
        self.pipes += [np.array([self.size-2, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step])]
        self.pipes += [np.array([self.size-2.5, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step-2])]
        self.pipes += [np.array([self.size-3, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, 8.5+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 4+step, 5.5+step]),
                       np.array([self.size-5, self.size-4.5, 5.5+step, 9+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0.5+step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 3.5+step, 4+step]), np.array([self.size-3, self.size-2.5, 9.5+step, 10+step]),
                       np.array([self.size-5, self.size-4.5, 2+step, 4+step]), np.array([self.size-3, self.size-2.5, 6.5+step, 7.5+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 1+step, 1.5+step]),
                       np.array([self.size-5, self.size-4.5, 1+step, 1.5+step]), np.array([self.size-3, self.size-2.5, 2.5+step, 3+step]),
                       np.array([self.size-3, self.size-2.5, 5+step, 5.5+step]), np.array([self.size-5, self.size-4.5, 6.5+step, 8+step])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 0.5+step, 1.5+step]), np.array([self.size-3, self.size-1, 6+step, 6.5+step]),
                       np.array([self.size-5, self.size-4.5, step, 2+step]), np.array([self.size-1.5, self.size-1, 3+step, 3.5+step]),
                       np.array([self.size-2, self.size-1, 3.5+step, 4+step]), np.array([self.size-2.5, self.size-1, 4+step, 4.5+step]),
                       np.array([self.size-3, self.size-1, 4.5+step, 5+step]), np.array([self.size-1.5, self.size-1, 7.5+step, 8+step]),
                       np.array([self.size-2, self.size-1, 7+step, 7.5+step]), np.array([self.size-2.5, self.size-1, 6.5+step, 7+step]),]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, 2.5+step]), np.array([self.size-1, self.size, 4+step, self.size+step])]
        self.stones += [np.array([self.size-1, self.size, step, 2.5+step]), np.array([self.size-1.5, self.size-1, step, 0.5+step]),
                       np.array([self.size-2, self.size-1, 0.5+step, 1+step]), np.array([self.size-2.5, self.size-1, 1+step, 1.5+step]),
                       np.array([self.size-3, self.size-1, 1.5+step, 2+step]), np.array([self.size-3, self.size-1, 2+step, 2.5+step]),
                       np.array([self.size-1.5, self.size-1, 5.5+step, 6+step]),
                       np.array([self.size-2, self.size-1, 5+step, 5.5+step]), np.array([self.size-2.5, self.size-1, 4.5+step, 5+step]),
                       np.array([self.size-3, self.size-1, 4+step, 4.5+step])]
        self.pipes += [np.array([self.size-2, self.size-1, step+8, step+9])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.pipes += [np.array([self.size-2, self.size-1, 6+step, 7+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 0.5+step, 2.5+step]), np.array([self.size-1.5, self.size-1, 7+step, 7.5+step]),
                       np.array([self.size-2, self.size-1, 7.5+step, 8+step]),np.array([self.size-2.5, self.size-1, 8+step, 8.5+step]),
                       np.array([self.size-3, self.size-1, 8.5+step, 9+step]),np.array([self.size-3.5, self.size-1, 9+step, 9.5+step]),
                       np.array([self.size-4, self.size-1, 9.5+step, 10+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-4.5, self.size-1, step, 0.5+step]),
                       np.array([self.size-5, self.size-1, 0.5+step, 1+step]),np.array([self.size-5.5, self.size-1, 1+step, 1.5+step]),
                       np.array([self.size-1.5, self.size-1, 9.5+step, 10+step])]
        
        self.flag = [np.array([3, self.size-1.5, 9+step, 10+step])]


        self.obstacles = self.ground + self.stones + self.pipes
        self.init_canvas()

        self.agent = pygame.image.load("./source/mario.png")
        self.agent = pygame.transform.scale(self.agent, (self.pix_square_size / 2, self.pix_square_size / 2))

        self.champi = pygame.image.load("./source/champi.png")
        self.champi = pygame.transform.scale(self.champi, (self.pix_square_size / 2, self.pix_square_size / 2))

        # Initialize the Enemies
        self.monsters = [Monster(self.size-1.6, 7, self.obstacles), Monster(self.size-1.6, self.size+7, self.obstacles), Monster(self.size-1.6, 2*self.size+7, self.obstacles), Monster(self.size-1.6, 2*self.size+7.5, self.obstacles),
                         Monster(self.size-5.6, 4*self.size+5.5, self.obstacles), Monster(self.size-5.6, 4*self.size+6.5, self.obstacles),
                         Monster(self.size-1.6, 7.5+5*self.size, self.obstacles), Monster(self.size-1.6, 7+5*self.size, self.obstacles),
                         Monster(self.size-1.6, 6*self.size, self.obstacles), Monster(self.size-1.6, 5+6*self.size, self.obstacles),
                         Monster(self.size-1.6, 5.5+6*self.size, self.obstacles),
                         Monster(self.size-1.6, 7*self.size, self.obstacles), Monster(self.size-1.6, 0.5+7*self.size, self.obstacles),
                         Monster(self.size-1.6, 2+7*self.size, self.obstacles), Monster(self.size-1.6, 2.5+7*self.size, self.obstacles),
                         Monster(self.size-1.6, 4.5+9*self.size, self.obstacles), Monster(self.size-1.6, 5+9*self.size, self.obstacles)]
        # self.monsters = []
    
    def _get_info(self):
        return {
            "distance": self.max_pos,
            "last_pos": self.last_loc,
            "x_pos": self._agent_location[0],
            "stage": 1
        }
    
    def update_position(self):
        self.top = self._agent_location[1] - 0.5
        self.bottom = self._agent_location[1]
        self.right = self._agent_location[0] + 0.5
        self.left = self._agent_location[0]

    def top_contact(self):
        for obstacle in self.obstacles:
            if self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]:
                if (self.right >= obstacle[2] and self.right < obstacle[3]):
                    self.vert_speed = obstacle[1] - self.top
                elif (self.left > obstacle[2] and self.left <= obstacle[3]):
                    self.vert_speed = obstacle[1] - self.top

    def bot_contact(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.right < obstacle[2] and self.right + self.horiz_speed >= obstacle[2]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left > obstacle[3] and self.left + self.horiz_speed <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                
    def can_jump(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    return True
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    return True
        return False

    def right_contact(self):
        for obstacle in self.obstacles:
            if self.right <= obstacle[2] and self.right + self.horiz_speed > obstacle[2]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[2] - self.right
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[2] - self.right
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.right
                elif (self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.right

    def left_contact(self):
        for obstacle in self.obstacles:
            if self.left >= obstacle[3] and self.left + self.horiz_speed < obstacle[3]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.bottom <= obstacle[1] and self.bottom + self.vert_speed > obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left


    def reset(self, seed=None, options=None):

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, self.size-2])
        self.max_pos = self._agent_location[0]
        self.last_loc = self._agent_location[0]
        self.update_position()
        self.terminated = False
        self.jump_count = 0
        for monster in self.monsters:
            monster.reset()

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        observation = self._render_frame()

        return observation


    def step(self, action):
        dvs = 10*0.04*self.ratio
        jump_state = self.can_jump()
        if action in [3, 5, 1]:
            if jump_state:
                self.vert_speed = -2.5 * self.ratio
                self.jump_count = 0
            dvs = dvs / 2
        if action in [0, 1] and (self.horiz_speed >= 0 or jump_state):
            self.horiz_speed = 1*self.ratio
        if action in [2, 5] and (self.horiz_speed <= 0 or jump_state):
            self.horiz_speed = -1*self.ratio
        if action == 4:
            self.horiz_speed = 0

        self.update_position()
        self.top_contact()
        self.bot_contact()
        self.right_contact()
        self.left_contact()

        for monster in self.monsters:
            monster.step(self)
            self.terminated = self.terminated or monster.player_left_contact(self) or monster.player_right_contact(self)
        
        self.horiz_speed = np.clip(self.horiz_speed, -1*self.ratio, 1*self.ratio)
        direction = np.array([1, 0])*self.horiz_speed + np.array([0, 1])*self.vert_speed

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, 200*self.size
        )
        self.vert_speed += dvs

        if self._agent_location[0] // 5 > self.max_pos // 5:
            reward = 0.1
        elif self._agent_location[0] <= self.last_loc:
            reward = -0.01
        else:
            reward =  0  # Binary sparse rewards
        observation = self._render_frame()
        info = self._get_info()

        self.max_pos = max(self.max_pos, self._agent_location[0])
        self.last_loc = self._agent_location[0]

        if self._agent_location[1] >= self.size:
            self.terminated = True

        if self._agent_location[0] >= 10*self.size + 9.5:
            info["flag_get"] = True
        else:
            info["flag_get"] = False
        
        if info["flag_get"]:
            reward = 0.2
            self.terminated = True

        return observation, reward, self.terminated, info


    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def init_canvas(self):
        i = 15
        canvas = pygame.Surface((i*self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        bg = pygame.image.load("./source/background.png")
        bg = pygame.transform.scale(bg, (self.window_size, self.window_size))
        for j in range(i):
            canvas.blit(bg, (j*self.window_size, 0))
        self.pix_square_size = (
            self.window_size / 8 #self.size
        )  # The size of a single grid square in pixels
        pix_square_size = self.pix_square_size

        for obstacle in self.flag:
            flag = pygame.image.load("./source/flag.png")
            flag = pygame.transform.scale(flag, (pix_square_size*(obstacle[3] - obstacle[2]), pix_square_size*(obstacle[1] - obstacle[0])))
            canvas.blit(flag, ((obstacle[2]+self.size/2)*pix_square_size, (obstacle[0]-2)*pix_square_size))

        ground = pygame.image.load("./source/ground.png")
        ground = pygame.transform.scale(ground, (pix_square_size / 2, pix_square_size / 2))

        for obstacle in self.pre_obs:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(ground, ((l+self.size/2)*pix_square_size, (h-2)*pix_square_size))


        for obstacle in self.ground:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(ground, ((l+self.size/2)*pix_square_size, (h-2)*pix_square_size))

        stone = pygame.image.load("./source/stone.png")
        stone = pygame.transform.scale(stone, (pix_square_size / 2, pix_square_size / 2))
        for obstacle in self.stones:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(stone, ((l+self.size/2)*pix_square_size, (h-2)*pix_square_size))

        for obstacle in self.pipes:
            pipe = pygame.image.load("./source/pipes.png")
            pipe = pygame.transform.scale(pipe, (pix_square_size*(obstacle[3] - obstacle[2]), pix_square_size*(obstacle[1] - obstacle[0])))
            canvas.blit(pipe, ((obstacle[2]+self.size/2)*pix_square_size, (obstacle[0]-2)*pix_square_size))

        self.in_canvas = canvas


    def _render_frame(self, have_state=False):

        pix_square_size = (
            self.window_size / 8 #self.size
        )  # The size of a single grid square in pixels

        back_screen = self.in_canvas.copy()
        rect = pygame.Rect(pix_square_size*(self._agent_location[0]+1), 0, self.window_size, self.window_size)
        try:
            sub = back_screen.subsurface(rect)
        except:
            print(pix_square_size*(self._agent_location[0]+1))
            print(15*self.window_size)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.blit(sub, (0, 0))
        # Now we draw the agent
        if self.horiz_speed < 0:
            agent = pygame.transform.flip(self.agent, flip_x=True, flip_y=False)
        else:
            agent = self.agent
        canvas.blit(agent, (pix_square_size*(self.size/2-1), pix_square_size*(self._agent_location[1]-2.5)))


        for monster in self.monsters:
            canvas.blit(self.champi, ((monster.left - 1 - self._agent_location[0] + self.size/2)*pix_square_size, (monster.top-2)*pix_square_size))
            
        self.canvas = canvas

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



class Monster():
    def __init__(self, top_pos, left_pos, obstacles):
        self.monster_pos = np.array([left_pos, top_pos])
        self.horiz_speed = -0.08
        self.last_speed = -0.08
        self.vert_speed = 0
        self.obstacles = obstacles
        self.contact = False
        self.init_pos = np.array([left_pos, top_pos])
        self.update_position()
        self.is_dead = False

    def update_position(self):
        self.top = self.monster_pos[1]
        self.bottom = self.monster_pos[1] + 0.5
        self.right = self.monster_pos[0] + 0.5
        self.left = self.monster_pos[0]

    def bot_contact(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.right < obstacle[2] and self.right + self.horiz_speed >= obstacle[2]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left > obstacle[3] and self.left + self.horiz_speed <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)

    def right_contact(self):
        for obstacle in self.obstacles:
            if self.right <= obstacle[2] and self.right + self.horiz_speed > obstacle[2]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[2] - self.right
                    self.contact = True
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[2] - self.right
                    self.contact = True
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.right
                    self.contact = True
                elif (self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.right
                    self.contact = True

    def left_contact(self):
        for obstacle in self.obstacles:
            if self.left >= obstacle[3] and self.left + self.horiz_speed < obstacle[3]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.bottom <= obstacle[1] and self.bottom + self.vert_speed > obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
    
    def reset(self):
        self.monster_pos = np.copy(self.init_pos)
        self.is_dead = False
        self.horiz_speed = -0.08
        self.last_speed = -0.08
        self.contact = False
        self.update_position()

    def can_jump(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    return True
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    return True
        return False

    def step(self, agent):
        if not self.is_dead and np.abs(agent.right - self.left) < agent.size:
            self.bot_contact()
            self.right_contact()
            self.left_contact()
            direction = np.array([1, 0])*self.horiz_speed + np.array([0, 1])*self.vert_speed
            self.monster_pos += direction
            self.update_position()
            self.vert_speed += 10*0.06/3

            self.is_dead = self.player_top_contact(agent)
            if self.monster_pos[1]+0.5 >= 10:
                self.is_dead = True

            if self.contact:
                self.horiz_speed = -1*self.last_speed
                self.last_speed *= -1
                self.contact = False
            
            if self.is_dead:
                self.monster_pos = np.array([-1, -1])
                self.update_position()

    def player_top_contact(self, player):
        if player.bottom <= self.top and player.bottom + player.vert_speed > self.top:
            if (player.right > self.left and player.right <= self.right):
                return True
            elif (player.left >= self.left and player.left < self.right):
                return True
            elif (player.right < self.left and player.right + player.horiz_speed >= self.left):
                return True
            elif (player.left > self.right and player.left + player.horiz_speed <= self.right):
                return True
            return False
        return False
    
    def player_right_contact(self, player):
        if player.right <= self.left and player.right + player.horiz_speed > self.left+self.horiz_speed:
            if (player.top <= self.bottom and player.top >= self.top):
                return True
            elif (player.bottom >= self.top and player.bottom <= self.bottom):
                return True
            elif (player.top >= self.bottom and player.top + player.vert_speed <= self.bottom):
                return True
            elif (player.bottom <= self.top and player.bottom + player.vert_speed >= self.top):
                return True
            return False
        return False

    def player_left_contact(self, player):
        if player.left >= self.right and player.left + player.horiz_speed < self.right+self.horiz_speed:
            if (player.top <= self.bottom and player.top >= self.top):
                return True
            elif (player.bottom >= self.top and player.bottom <= self.bottom):
                return True
            elif (player.top >= self.bottom and player.top + player.vert_speed <= self.bottom):
                return True
            elif (player.bottom <= self.top and player.bottom + player.vert_speed >= self.top):
                return True
            return False
        return False