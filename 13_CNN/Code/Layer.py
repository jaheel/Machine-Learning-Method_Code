from abc import abstractmethod

class Layer(object):
    @abstractmethod
    def forward_propagation(self, **kwargs):
        pass

    @abstractmethod
    def back_propagation(self, **kwargs):
        pass
