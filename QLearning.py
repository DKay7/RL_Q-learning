import random
import time
import os
import numpy as np
import pandas as pd


class Q:
    def __init__(self):
        self.gamma = 0.95
        self.alpha = 0.05
        self.agent = None
        self.q_table = {}
        self.q_table_data = None

    def set_agent_object(self, agent):
        self.agent = agent

    def get_q_table(self):
        return self.q_table

    def safe_q_table(self, time_=False, name='q_table_saved'):
        self.q_table_data = pd.DataFrame(self.q_table)
        self.q_table_data.values.astype(float)

        if time_:
            name = 'saved_data/'+name+'{0}.csv'.format(time.localtime(time.time()))
        else:
            name = 'saved_data/'+name+'.csv'

        self.q_table_data.to_csv(name, index=False)

    def read_q_table(self, name='q_table_saved'):
        name = 'saved_data/' + name + '.csv'

        self.q_table_data = pd.read_csv(name, header=[0, 1, 2, 3])
        self.q_table = {}

        for index, column in enumerate(self.q_table_data.columns.to_list()):

            self.q_table[tuple(map(int, list(column)))] = \
                self.q_table_data.values[:, index].tolist()

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def teaching(self, visualise=False):

        self.agent.previous_action = self.agent.current_action
        self.agent.previous_state = self.agent.current_state

        self.agent.current_action = self.agent.select_action()
        self.agent.current_state = self.agent.get_state(self.agent.x, self.agent.y)

        if visualise:
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

        super().__init__(x, y)

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
        #  состояние -- координаты всех врагов и величины,
        #  на которые хочет сдвинуться агент

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

        if (0 <= new_x < self.dim) and (0 <= new_y < self.dim):
            self.x = new_x
            self.y = new_y


class Enemy(Unit):
    def __init__(self, x, y, dim):
        super().__init__(x, y)
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

    def visualise(self, visualise=False):

        self.map = list([['=' for _ in range(self.dim)] for _ in range(self.dim)])

        agent_x, agent_y = self.agent.get_coordinates()
        self.map[agent_x][agent_y] = 'A'

        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.get_coordinates()
            self.map[enemy_x][enemy_y] = 'E'

        if visualise:
            os.system('cls')
            for row in self.map:
                print(*row)

    def is_finished(self):
        not_finished = False

        agent_x, agent_y = self.agent.get_coordinates()

        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.get_coordinates()

            not_finished = not_finished or ((agent_x, agent_y) == (enemy_x, enemy_y))

        return not_finished

    def get_reward(self, finished):
        if not finished:
            self.agent.reward = 1
        else:
            self.agent.reward = -1

    def play(self, visualise=False):

        finished = self.is_finished()

        iteration_ = 0

        while not finished:

            self.visualise(visualise)
            self.step()
            finished = self.is_finished()
            self.get_reward(finished)
            self.q_model.teaching(visualise=visualise)

            if visualise:
                print('___')
                time.sleep(0.8)

            iteration_ += 1

        return iteration_
