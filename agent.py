import multiprocessing
import numpy as np
import objective_function
import gym
import helper
import torch
import os
import glob
import network
from collections import OrderedDict
from operator import itemgetter
from abc import ABCMeta, abstractmethod


def run_worker(worker, worker_id, env, config, return_dict):
    """start worker evaluation based on the objective function"""
    obj_func = getattr(objective_function, config["objective_function"])
    return_dict[worker_id] = obj_func(worker, env, config["max_steps_in_episodes"], config["gamma"])


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, config):
        super()
        self.config = config
        self.create_network = getattr(network, config["Network"])
        self.model = self.create_network(config)
        self.best_weights = self.model.get_weights()
        self.objective_function = getattr(objective_function, config["objective_function"])

    @abstractmethod
    def learn(self):
        pass

    def act(self, state):
        """select one action based on the current state"""
        return self.model.forward(state)

    def save(self):
        """Save the network weights"""
        save_dir = os.path.join(".", *self.config["checkpoint_dir"], self.config["env_name"])
        helper.mkdir(save_dir)
        current_date_time = helper.get_current_date_time()
        current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")

        torch.save(self.model.state_dict(), os.path.join(save_dir, "ckpt_" + current_date_time))

    def load(self):
        """Load latest available network weights"""
        load_path = os.path.join(".", *self.config["checkpoint_dir"], self.config["env_name"], "*")
        list_of_files = glob.glob(load_path)
        latest_file = max(list_of_files, key=os.path.getctime)
        self.model.load_state_dict(torch.load(latest_file))


class HillClimbing(BaseAgent):
    """Agent that uses Hill Climbing to learn"""
    def __init__(self, config):
        super().__init__(config)
        self.best_score = self.objective_function(self.model,
                                                  gym.make(self.config["env_name"]),
                                                  self.config["max_steps_in_episodes"],
                                                  1.0)

    def learn(self):
        """run one learning step"""
        new_weights = [w + self.config["sigma"] * np.random.normal(size=w.shape) for w in self.best_weights]
        self.model.set_weights(new_weights)
        new_score = self.objective_function(self.model,
                                            gym.make(self.config["env_name"]),
                                            self.config["max_steps_in_episodes"],
                                            self.config["gamma"])

        if new_score > self.best_score:
            self.best_weights = new_weights

        self.model.set_weights(self.best_weights)
        self.best_score = self.objective_function(self.model, gym.make(self.config["env_name"]),
                                                  self.config["max_steps_in_episodes"], 1.0)
        return self.best_score

