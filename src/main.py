import tensorflow as tf
import numpy as np
from keras.activations import sigmoid
import keras
from random import Random

HEIGHT = 10
WIDTH = 10


class Maze:
    def __init__(self):
        self.maze = [
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        ]
        self.width = len(self.maze[0])
        self.height = len(self.maze)
        self.x = 0
        self.y = 0

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def move(self, index):
        x = self.x
        y = self.y
        if index == 0:
            y += 1
        elif index == 1:
            y -= 1
        elif index == 2:
            x += 1
        elif index == 3:
            x -= 1
        if (
            x < 0
            or x >= self.width
            or y < 0
            or y >= self.height
            or self.maze[y][x] == 1
        ):
            return
        self.x = x
        self.y = y

    def observations(self):
        observations = []
        for row in self.maze:
            observations.extend(row)
        observations.extend([1 if i == self.x else 0 for i in range(self.width)])
        observations.extend([1 if i == self.y else 0 for i in range(self.height)])
        return observations

    def is_solved(self):
        return self.x == self.width - 1 and self.y == self.height - 1

    def reward(self):
        dx = abs(self.x - self.width + 1)
        dy = abs(self.y - self.height + 1)
        return 1 / (dx + dy)


INPUT_SIZE = len(Maze().observations())
OUTPUT_SIZE = 4  # Up, Down, Left, Right


@tf.function
def feed_forward(inputs, layers):
    variables = inputs
    for weights, biases in layers:
        variables = variables @ weights
        variables = variables + biases
        variables = sigmoid(variables)
    return variables


@tf.function
def feed_forward_argmax(inputs, layers):
    outputs = feed_forward(inputs, layers)
    return tf.argmax(outputs, 1)


@tf.function
def train(state_1, choices, rewards, state_2, network, target, optimizer, gamma):
    train_variables = []

    with tf.GradientTape() as tape:

        for weights, biases in network:
            train_variables.extend([weights, biases])

        network_out = feed_forward(state_1, network)
        network_var_out = tf.gather(network_out, choices, batch_dims=1)

        target_out = feed_forward(state_2, target)
        target_out_max = tf.reduce_max(target_out, axis=1)
        target_out_scaled = tf.multiply(gamma, target_out_max)

        loss_raw = target_out_scaled - network_var_out - rewards

        loss = tf.math.square(loss_raw)
        loss_mean = tf.reduce_mean(loss)
        gradient = tape.gradient(loss_mean, train_variables)
        optimizer.apply_gradients(zip(gradient, train_variables))


class Agent:
    def __init__(self, layer_sizes: list[int]):
        layer_sizes.append(4)
        self.network = []
        self.target = []
        self.replay = []
        self.iter = 0
        self.random = Random()
        self.max_replay = 10_000

        prev = INPUT_SIZE

        for layer in layer_sizes:
            weights = tf.random.normal((prev, layer))
            bias = tf.random.normal((layer,))
            self.network.append((weights, bias))
            prev = layer
        self.update_target()

    def update_target(self):
        self.target = [
            (tf.constant(weights), tf.constant(biases))
            for weights, biases in self.network
        ]

    def generate_maze(self):
        return Maze()

    def epsilon(self):
        return 0.2

    def learning_rate(self):
        return 0.1

    def gamma(self):
        return 0.5

    def populate_replay(self, count: int):
        maze = self.generate_maze()
        for _ in range(count):
            if maze.is_solved():
                maze = self.generate_maze()

            obs = maze.observations()

            sel = self.random.randrange(0, 4)
            if self.epsilon() < self.random.random():
                obs_tf = tf.constant(obs, dtype=tf.float32, shape=(1, len(obs)))
                sel = int(feed_forward_argmax(obs_tf, self.network))

            maze.move(sel)
            obs_2 = maze.observations()

            reward = maze.reward()

            l = len(self.replay)
            if l >= self.max_replay:
                self.replay.remove(self.random.randint(0, l))

            self.replay.append((obs, sel, obs_2, reward))

        l = len(self.replay)

        if l > self.max_replay:
            indexes = self.random.sample(range(l), k=self.max_replay - l)
            indexes.sort()
            for i, index in enumerate(indexes):
                self.replay.remove(index - i)

    def train_replay(self, count: int):
        self.iter += 1

        values = self.random.choices(self.replay, k=count)
        state_1 = []
        choices = []
        rewards = []
        state_2 = []

        for st_1, choice, st_2, reward in values:
            state_1.append(st_1)
            state_2.append(st_2)
            choices.append(choice)
            rewards.append(reward)

        state_1_tf = tf.constant(state_1, dtype=tf.float32)
        state_2_tf = tf.constant(state_2, dtype=tf.float32)
        choices_tf = tf.constant(choices)
        rewards_tf = tf.constant(rewards, dtype=tf.float32)

        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate())

        train(
            state_1_tf,
            choices_tf,
            rewards_tf,
            state_2_tf,
            self.network,
            self.target,
            optimizer,
            self.gamma(),
        )


class ModifiedAgent(Agent):
    def generate_maze(self):
        return Maze()


if __name__ == "__main__":
    agent = Agent([9, 8, 7, 6, 5])
    agent.populate_replay(100)
    agent.train_replay(5)
