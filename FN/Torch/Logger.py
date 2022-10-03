import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplt as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir = "", dir_name = "") -> None:
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
                
        self._callback = SummaryWriter(log_dir=log_dir)
        
    @property
    def writer(self):
        return self._callback.file_writer
    
    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)
    
    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))
            
    def plot(self, name, values, interval = 10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.arrray(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha = 0.1 , color = "g")
        plt.plot(indices, means, "o-", color="g", label = "{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()
        
    def write(self, index, name, value):
        summary = SummaryWriter()
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = value
        self.writer.add_summary(summary ,index)
        self.writer.flush()
        
        