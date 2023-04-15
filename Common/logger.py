import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import glob
from collections import deque


class Logger:
    def __init__(self, agent, **config):
        self.config = config
        # self.experiment = self.config["experiment"]
        self.agent = agent
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_ext_reward = 0
        self.running_ext_reward = 0
        self.running_int_reward = 0
        self.running_act_prob = 0
        self.running_training_logs = dict()
        self.x_pos = 0
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_ext_r = 0  # It is not correct but does not matter.
        self.max_episode_reward = 0

        if self.config["do_train"] and self.config["train_from_scratch"]:
            self.create_wights_folder()

        self.exp_avg = lambda x, y: 0.99 * x + 0.01 * y if y != 0 else y

    def create_wights_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        algo = self.config["algo"]
        os.mkdir(f"Models/{algo}/" + self.log_dir)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, training_logs, int_reward, action_prob = args

        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)
        if iteration == 1:
            for k, v in training_logs.items():
                self.running_training_logs.update({k: v})
        else:
            for k, v in training_logs.items():
                self.running_training_logs[k] = self.exp_avg(self.running_training_logs[k], v)

        if iteration % self.config["interval"] == 0:
            self.save_params(self.episode, iteration)

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Episode Ext Reward", self.episode_ext_reward, self.episode)
            writer.add_scalar("Running Episode Ext Reward", self.running_ext_reward, self.episode)
            writer.add_scalar("Position", self.x_pos, self.episode)
            writer.add_scalar("Running last 10 Ext Reward", self.running_last_10_ext_r, self.episode)
            writer.add_scalar("Max Episode Ext Reward", self.max_episode_reward, self.episode)
            writer.add_scalar("Running Action Probability", self.running_act_prob, iteration)
            writer.add_scalar("Running Intrinsic Reward", self.running_int_reward, iteration)
            writer.add_scalar("Running PG Loss", self.running_training_logs["pg_loss"], iteration)
            writer.add_scalar("Running Ext Value Loss", self.running_training_logs["ext_value_loss"], iteration)
            writer.add_scalar("Running Int Value Loss", self.running_training_logs["int_value_loss"], iteration)
            writer.add_scalar("Running RND Loss", self.running_training_logs["rnd_loss"], iteration)
            writer.add_scalar("Running Entropy", self.running_training_logs["entropy"], iteration)
            writer.add_scalar("Running Intrinsic Explained variance", self.running_training_logs["int_ep"], iteration)
            writer.add_scalar("Running Extrinsic Explained variance", self.running_training_logs["grad_norm"], iteration)

        self.off()
        if iteration % self.config["interval"] == 0:
            print("Iter: {}| "
                  "EP: {}| "
                  "EP_Reward: {}| "
                  "EP_Running_Reward: {:.2f}| "
                  "Position: {:.1f}| "
                  "Iter_Duration: {:.3f}| "
                  "Time: {} "
                  .format(iteration,
                          self.episode,
                          self.episode_ext_reward,
                          self.running_ext_reward,
                          self.x_pos,
                          self.duration,
                          datetime.datetime.now().strftime("%H:%M:%S")
                          )
                  )
        self.on()

    def log_iteration_ddql(self, *args):
        iteration, training_logs, action_prob, explo_rate = args

        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        if iteration == 1:
            for k, v in training_logs.items():
                self.running_training_logs.update({k: v})
        else:
            for k, v in training_logs.items():
                self.running_training_logs[k] = self.exp_avg(self.running_training_logs[k], v)

        if iteration % self.config["interval"] == 0:
            self.save_params_ddql(self.episode, iteration)

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Episode Ext Reward", self.episode_ext_reward, self.episode)
            writer.add_scalar("Running Episode Ext Reward", self.running_ext_reward, self.episode)
            writer.add_scalar("Position", self.x_pos, self.episode)
            writer.add_scalar("Running last 10 Ext Reward", self.running_last_10_ext_r, self.episode)
            writer.add_scalar("Max Episode Ext Reward", self.max_episode_reward, self.episode)
            writer.add_scalar("Running Action Probability", self.running_act_prob, iteration)
            writer.add_scalar("Running Loss", self.running_training_logs["loss"], iteration)
            writer.add_scalar("Running TD estimation", self.running_training_logs["td_estimation"], iteration)
            writer.add_scalar("Running TD target", self.running_training_logs["td_target"], iteration)

        self.off()
        if iteration % self.config["interval"] == 0:
            print("Iter: {}| "
                  "EP: {}| "
                  "Exp_rate: {:.3f}| "
                  "EP_Reward: {}| "
                  "EP_Running_Reward: {:.2f}| "
                  "Position: {:.1f}| "
                  "Iter_Duration: {:.3f}| "
                  "Time: {} "
                  .format(iteration,
                          self.episode,
                          explo_rate,
                          self.episode_ext_reward,
                          self.running_ext_reward,
                          self.x_pos,
                          self.duration,
                          datetime.datetime.now().strftime("%H:%M:%S")
                          )
                  )
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_ext_reward, x_pos, visited_stages = args

        self.max_episode_reward = max(self.max_episode_reward, self.episode_ext_reward)
        self.running_ext_reward = self.exp_avg(self.running_ext_reward, self.episode_ext_reward)
        self.x_pos = self.exp_avg(self.x_pos, x_pos)

        self.last_10_ep_rewards.append(self.episode_ext_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_ext_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    def save_params_ddql(self, episode, iteration):
        torch.save({"model": self.agent.net.state_dict(),
                    "optimizer": self.agent.optimizer.state_dict(),
                    "iteration": iteration,
                    "episode": episode,
                    "running_ext_reward": self.running_ext_reward,
                    "running_act_prob": self.running_act_prob,
                    "running_training_logs": self.running_training_logs,
                    "x_pos": self.x_pos,
                    "exploration_rate": self.agent.exploration_rate,
                    },
                   "Models/DDQL/" + self.log_dir + "/params.pth")


    def save_params(self, episode, iteration):
        torch.save({"policy_state_dict": self.agent.policy.state_dict(),
                    "predictor_model_state_dict": self.agent.predictor_model.state_dict(),
                    "target_model_state_dict": self.agent.target_model.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "state_rms_mean": self.agent.state_rms.mean,
                    "state_rms_var": self.agent.state_rms.var,
                    "state_rms_count": self.agent.state_rms.count,
                    "int_reward_rms_mean": self.agent.int_reward_rms.mean,
                    "int_reward_rms_var": self.agent.int_reward_rms.var,
                    "int_reward_rms_count": self.agent.int_reward_rms.count,
                    "iteration": iteration,
                    "episode": episode,
                    "running_ext_reward": self.running_ext_reward,
                    "running_int_reward": self.running_int_reward,
                    "running_act_prob": self.running_act_prob,
                    "running_training_logs": self.running_training_logs,
                    "x_pos": self.x_pos,
                    },
                   "Models/RND/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/RND/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")

        self.agent.set_from_checkpoint(checkpoint)
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        self.running_ext_reward = checkpoint["running_ext_reward"]
        self.x_pos = checkpoint["x_pos"]
        self.episode = checkpoint["episode"]
        self.running_training_logs = checkpoint["running_training_logs"]
        self.running_act_prob = checkpoint["running_act_prob"]
        self.running_int_reward = checkpoint["running_int_reward"]

        return checkpoint["iteration"], self.episode

    def load_weights_ddql(self):
        model_dir = glob.glob("Models/DDQL/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.agent.load(checkpoint)
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        self.running_ext_reward = checkpoint["running_ext_reward"]
        self.x_pos = checkpoint["x_pos"]
        self.episode = checkpoint["episode"]
        self.running_training_logs = checkpoint["running_training_logs"]
        self.running_act_prob = checkpoint["running_act_prob"]

        return checkpoint["iteration"], self.episode