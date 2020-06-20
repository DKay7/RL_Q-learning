import random
import time
import os


class Q:
    def __init__(self):
        self.gamma = 0.95
        self.alpha = 0.05
        self.agent = None
        self.q_table = {}

    def set_agent_object(self, agent):
        self.agent = agent

    def teaching(self, silent=1):

        self.agent.previous_action = self.agent.current_action
        self.agent.previous_state = self.agent.current_state

        self.agent.current_action = self.agent.select_action()
        self.agent.current_state = self.agent.get_state(self.agent.x, self.agent.y)

        if not silent:
            print(self.agent.previous_state)
            print(self.agent.current_state)

        if self.agent.previous_state not in self.q_table:
            self.q_table[self.agent.previous_state] = [0 for _ in self.agent.actions]

        q_max = max(self.q_table[self.agent.previous_state])

        self.q_table[self.agent.previous_state][self.agent.previous_action] += \
            self.alpha * (self.agent.reward + self.gamma * q_max -
                          self.q_table[self.agent.previous_state][self.agent.previous_action]
                          )


class Unit:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                        (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_coordinates(self):
        return self.x, self.y


class Agent(Unit):

    def __init__(self, q_function, x, y, dim, enemies):

        Unit.__init__(self, x, y)

        self.dim = dim
        self.enemies = enemies
        self.q_model = q_function
        self.dx = 0
        self.dy = 0
        self.epsilon = 0.95
        self.reward = 0

        self.current_state = self.get_state(x, y)
        self.current_action = self.select_action()

        self.previous_state = self.get_state(x, y)
        self.previous_action = self.current_action

    def get_state(self, x, y):
        #  состояние -- координаты всех врагов

        features = []

        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.get_coordinates()

            features.append(enemy_x)
            features.append(enemy_y)

        features.append(x)
        features.append(y)

        state = tuple(features)

        return state

    def select_action(self):

        if random.random() < self.epsilon:

            action = random.choice([i_ for i_ in range(len(self.actions))])

        else:

            if self.current_state not in self.q_model.q_table:
                self.q_model.q_table[self.current_state] = [0 for _ in self.actions]

            action = max(list(enumerate(self.q_model.q_table[self.current_state])), key=lambda x: x[1])[0]

        return action

    def move(self):
        self.dx, self.dy = self.actions[self.select_action()]

        new_x = self.x + self.dx
        new_y = self.y + self.dy

        if (0 < new_x < self.dim) and (0 < new_y < self.dim):
            self.x = new_x
            self.y = new_y


class Enemy(Unit):
    def __init__(self, x, y, dim):
        Unit.__init__(self, x, y)
        self.dim = dim

    def move(self):
        expr = False

        while not expr:

            act = random.choice(self.actions)
            new_x = self.x + act[0]
            new_y = self.y + act[1]

            expr = ((0 <= new_x < self.dim) and (0 <= new_y < self.dim))

            if expr:
                self.x = new_x
                self.y = new_y


class Environment:

    def __init__(self, dim, q_function):
        self.dim = dim

        self.q_model = q_function
        self.enemies = [Enemy(dim - 2, dim - 2, dim)]
        self.agent = Agent(self.q_model, 1, 1, dim,  self.enemies)
        self.q_model.set_agent_object(self.agent)
        self.map = []

    def step(self):

        self.agent.move()

        for enemy in self.enemies:
            enemy.move()

    def visualise(self, silent=1):

        agent_x, agent_y = self.agent.get_coordinates()

        self.map = list([['=' for _ in range(self.dim)] for _ in range(self.dim)])
        self.map[agent_x][agent_y] = 'A'

        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.get_coordinates()
            self.map[enemy_x][enemy_y] = 'E'

        if not silent:
            os.system('cls')
            for row in self.map:
                print(*row)

    def is_finished(self):
        not_finished = True

        agent_x, agent_y = self.agent.get_coordinates()

        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.get_coordinates()

            not_finished = not_finished and ((agent_x, agent_y) != (enemy_x, enemy_y))

        return not_finished

    def get_reward(self, not_finished):

        if not_finished:
            self.agent.reward = 10

        else:
            self.agent.reward = -5

    def play(self, silent=1, silent_run=1):

        not_finished = self.is_finished()
        iteration_ = 0

        while not_finished:

            self.visualise(silent)
            self.step()
            not_finished = self.is_finished()
            self.get_reward(not_finished)

            if silent_run:
                self.q_model.teaching(silent)

            if not silent:
                print('___')
                time.sleep(0.8)

            iteration_ += 1

        return iteration_


if __name__ == '__main__':
    q_model = Q()
    iterations = []
    first_epochs = 500
    second_epochs = 1500
    test_epochs = 100
    dimension = 25
    max1, max2 = 0, 0
    for epoch in range(first_epochs):
        os.system('cls')
        print('#' * (100 * epoch // first_epochs))
        environment = Environment(dimension, q_model)
        environment.agent.epsilon = 0.90
        iteration = environment.play(1)
        environment.visualise(1)
        iterations.append(iteration)

    print('1st learning Epoch.\nMaximum living time {0}\n\n'.format(max(iterations)))
    time.sleep(5)

    max1 = max(iterations)

    iterations = []
    for epoch in range(second_epochs):
        os.system('cls')
        print('#' * (100 * epoch // second_epochs))
        environment = Environment(dimension, q_model)
        environment.agent.epsilon = 0.2
        iteration = environment.play(1)
        environment.visualise(1)
        iterations.append(iteration)

    print('2nd learning Epoch.\nMaximum living time {0}\n\n'.format(max(iterations)))
    time.sleep(5)

    max2 = max(iterations)

    print('___')
    iterations = []
    for i in range(100):
        os.system('cls')
        print(i, '\n\n')
        environment = Environment(dimension, q_model)
        environment.agent.epsilon = 0.0
        iteration = environment.play(1)
        environment.visualise(1)
        iterations.append(iteration)
        print('{0} GAME FINISHED WITH THE MAX TIME {1}'.format(i, max(iterations)))
        print('GONNA START NEW GAME')

    print('1st epoch max {0}'.format(max1))
    print('2nd epoch max {0}'.format(max2))
    print('LAST GAME FINISHED WITH THE MAX TIME {0}\n MEAN TIME {1}'.format(max(iterations),
                                                                            sum(iterations) / len(iterations)))
