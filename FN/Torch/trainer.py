from collections import deque
import re
from typing import Deque, List
from unittest import expectedFailure
from FN.TF.fn_framework import Experience
from fn_agent import FNAgent

class Trainer:
    buffer_size: int
    batch_size: int
    gamma: float
    report_interval: int
    log_dir: str
    experiences: Deque[Experience]
    training: bool
    training_count: int
    reward_log: List
    
    def __init__(
        self, 
        buffer_size = 1024,
        batch_size = 32,
        gamma = 0.9,
        report_interval = 10,
        log_dir = ""
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.log_dir = log_dir
        # Experience Replay
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        
    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked
    
    def train_loop(
        self,
        env,
        agent: FNAgent,
        episode = 200,
        initial_count = 1,
        render = False,
        observe_interval = 0        
    ):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []
        
        for i in range(episode):
            state = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            # エピソードが終わるまで繰り返す
            while not done:
                if render:
                    env.render()
                
                can_observe = (self.training_count == 1 or self.training_count % observe_interval == 0)
                if self.training and observe_interval > 0 and can_observe:
                    frames.append(state)
                
                action = agent.policy(state)
                n_state, reward, done, info = env.step(action)
                experience = Experience(state, action, reward, n_state, done)
                self.experiences.append(experience)
                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True
                    
                self.step(i, step_count, agent, experience)
                state = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent, experience)
                
                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True
                    
                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(
                            self.training_count,
                            frames
                        )
                        frames = []
                    self.training_count += 1
                    
    def episode_begin(self, episode, agent):
        pass
    
    def begin_train(self, episode, agent):
        pass
    
    def step(self, episode, step_count, agent, experience):
        pass                
                
    def episode_end(self, episode, step_count, agent):
        pass
    
    def is_event(self, count, interval):
        return (count != 0 and count % interval == 0)
    
    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]