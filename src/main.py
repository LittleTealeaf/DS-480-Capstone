import tensorflow as tf
from keras.activations import relu


HEIGHT = 10
WIDTH = 10


INPUT_SIZE = 10
OUTPUT_SIZE = 4  # Up, Down, Left, Right


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
        if x < 0 or x >= 10 or y < 0 or y >= 10 or self.maze[y][x] == 1:
            return
        self.x = x
        self.y = y


@tf.function
def feed_forward(inputs, layers):
    variables = inputs
    for weights, biases in layers:
        variables = variables @ weights
        variables = variables + biases
        variables = relu(variables)
    return variables


class Agent:
    def __init__(self, layer_sizes: list[int]):
        layer_sizes.append(4)
        self.network = []
        self.replay = []

        prev = INPUT_SIZE

        for layer in layer_sizes:
            weights = tf.random.normal((prev, layer))
            bias = tf.random.normal((layer,))
            self.network.append((weights, bias))
            prev = layer

    def collect_data(self):
        print(feed_forward(tf.random.normal((1, INPUT_SIZE)), self.network))


if __name__ == "__main__":
    agent = Agent([9, 8, 7, 6, 5])
    agent.collect_data()
