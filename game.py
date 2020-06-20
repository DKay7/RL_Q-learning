from QLearning import Q
from QLearning import Environment
import os
from tqdm import tqdm
from colorama import init


class Game:
    def __init__(self):

        init()

        self.epoch_iterations = []
        self.iterations = []
        self.iteration = 0
        self.epochs = 1

        self.q = Q()

        self.times = 5000

        self.test_epochs = 100
        self.dimension = 10

    def set_epochs(self, epochs=2, test_epochs=100, times=5000):
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.times = times

    def get_data(self):
        data = {
            'lifetime': iterations,
            'epoches': self.epochs,
            'times': self.times,
            'mean': [sum(self.iterations[i]) / len(self.iterations[i]) for i in self.iterations],
            'max': [max(self.iterations[i]) for i in self.iterations],
            'min': [min(self.iterations[i]) for i in self.iterations]
        }
        return data

    def train(self):

        for epoch in tqdm(range(self.epochs), position=0, leave=False):

            self.epoch_iterations = []

            for _ in tqdm(range(self.times)):
                environment = Environment(self.dimension, self.q)
                environment.agent.epsilon = 1 - ((epoch+1) * (0.7 / 10*(self.epochs+1)))

                self.iteration = environment.play(visualise=False)
                environment.visualise(visualise=False)

                self.epoch_iterations.append(self.iteration)

            self.iterations.append(self.epoch_iterations)

    def test(self):

        self.epoch_iterations = []

        for i in range(self.test_epochs):
            environment = Environment(self.dimension, self.q)
            environment.agent.epsilon = 0.0
            self.iteration = environment.play()
            environment.visualise()
            self.epoch_iterations.append(self.iteration)

        self.iterations.append(self.epoch_iterations)

        print('Epochs maxes: {0}'.format(max(self.iterations)))
        print('LAST GAME FINISHED WITH THE MAX TIME {0}\n MEAN TIME {1}'.format(max(self.iterations[-1]),
                                                                                sum(self.iterations[-1]) /
                                                                                len(self.iterations[-1])))


game_epochs = Game()
game_epochs.set_epochs(epochs=3, times=100000, test_epochs=5000)

game_times = Game()
game_times.set_epochs(epochs=3, times=100000, test_epochs=5000)

# game.train()
# q_t = game.q.get_q_table()
# game.q.safe_q_table(name='more_times_less_epochs')
# game.test()

game_epochs.q.read_q_table(name='more_epochs_less_times')
game_epochs.test()

game_times.q.read_q_table(name='more_times_less_epochs')
game_times.test()
