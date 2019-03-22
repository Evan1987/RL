import abc


class RL(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def choose_action(self, observation):
        pass

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        pass
