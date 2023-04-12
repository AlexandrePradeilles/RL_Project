import os
from .utils import *
import time


class Play:
    def __init__(self, env, agent, max_episode=1):
        if env is not None:
            self.env = make_mario(env, 1500, sticky_action=False)
        else:
            self.env = make_handmade_mario(1500, sticky_action=False)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "result" + ".avi", self.fourcc, 20.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((4, 84, 84), dtype=np.uint8)
        for ep in range(self.max_episode):
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            done = False
            while not done:
                action, *_ = self.agent.get_actions_and_values(stacked_states)
                s_, r, done, info = self.env.step(action[0])
                if info["flag_get"]:
                    print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩")
                episode_reward += r

                stacked_states = stack_states(stacked_states, s_, False)

                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward}| "
                  f"pos:{info['x_pos']}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()


class PlayDDQL:
    def __init__(self, env, agent, max_episode=1):
        if env is not None:
            self.env = make_mario(env, 1500, sticky_action=False)
        else:
            self.env = make_handmade_mario(1500, sticky_action=False)
        self.max_episode = max_episode
        self.agent = agent
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "result" + ".avi", self.fourcc, 20.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((4, 84, 84), dtype=np.uint8)
        for ep in range(self.max_episode):
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            done = False
            while not done:
                action, *_ = self.agent.step(stacked_states)
                s_, r, done, info = self.env.step(action)
                if info["flag_get"]:
                    print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩")
                episode_reward += r

                stacked_states = stack_states(stacked_states, s_, False)

                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward}| "
                  f"pos:{info['x_pos']}")
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()