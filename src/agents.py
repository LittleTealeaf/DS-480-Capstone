from environment import Environment
import tensorflow as tf
from random import Random
from keras.activations import sigmoid, relu
from keras.optimizers import SGD


def has_invalid_numbers(tensor):
    return tf.reduce_any(tf.logical_or(tf.math.is_inf(tensor), tf.math.is_nan(tensor)))


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
def train(state_1, choices, rewards, state_2, network, target, gamma, optimizer):
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
    def __init__(
        self,
        layer_sizes: list[int],
        max_replay=10_000,
        height=10,
        width=10,
        seed=None,
        target_update_frequency=100,
        updates_per_step=1,
    ):
        layer_sizes.append(4)
        self.network = []
        self.target = []
        self.replay = []
        self.iter = 0
        self.updates_per_step = updates_per_step
        self.random = Random(seed)
        self.max_replay = max_replay
        self.height = height
        self.width = width
        self.obs_size = Environment(height, width).get_obs_length()
        self.optimizer = None
        self.target_update_frequency = target_update_frequency

        prev = self.obs_size
        for layer in layer_sizes:
            weights = tf.Variable(tf.random.normal((prev, layer)))
            bias = tf.Variable(tf.random.normal((layer,)))
            self.network.append((weights, bias))
            prev = layer
        self.update_target()

    def update_target(self):
        self.target = [
            (tf.stop_gradient(weights), tf.stop_gradient(bias))
            for weights, bias in self.network
        ]

    def learning_rate(self):
        return 0.1 * (
            0.8 ** (self.iter // self.target_update_frequency)
            * 0.9 ** (self.iter % self.target_update_frequency)
        )

    def gamma(self):
        return 0.9

    def epsilon(self):
        return (
            0.5
            * (0.9 ** (self.iter // self.target_update_frequency))
            * (0.9 ** (self.iter % self.target_update_frequency))
            + 0.3
        )

    def new_environment(self) -> Environment:
        return Environment(
            self.height,
            self.width,
            seed=self.random.randint(0, 2**23 - 1),
        )

    def create_environment(self) -> Environment:
        return self.new_environment()

    def populate_replay(self, count: int):
        env = self.create_environment()
        for _ in range(count):
            while env.is_solved():
                env = self.create_environment()

            obs = env.get_observations()
            sel = self.random.randrange(0, 4)
            if self.epsilon() < self.random.random():
                obs_tf = tf.constant(obs, dtype=tf.float32, shape=(1, obs.size))
                sel = int(feed_forward_argmax(obs_tf, self.network))

            env.move(sel)
            obs_2 = env.get_observations()

            reward = env.get_reward()

            if len(self.replay) > self.max_replay:
                index = self.random.randint(0, self.max_replay)
                self.replay[index] = (obs, sel, obs_2, reward)
            else:
                self.replay.append((obs, sel, obs_2, reward))

    def train(self, count: int):
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

        if self.optimizer is None:
            self.optimizer = SGD(self.learning_rate())
        else:
            self.optimizer.learning_rate = self.learning_rate()

        train(
            state_1_tf,
            choices_tf,
            rewards_tf,
            state_2_tf,
            self.network,
            self.target,
            self.gamma(),
            self.optimizer,
        )

        if self.iter % self.target_update_frequency == 0:
            self.update_target()

    def evaluate(self, k: int = 1):
        envs = [self.create_environment() for _ in range(k)]

        len_obs = envs[0].get_obs_length()

        positions = []
        observations = []

        for env in envs:
            pos = env.get_valid_positions()
            positions.append(pos)
            for x, y in pos:
                env.set_position(x, y)
                observations.append(env.get_observations())

        observations_tf = tf.constant(
            observations, shape=(len(observations), len_obs), dtype=tf.float32
        )

        choices = feed_forward_argmax(observations_tf, self.network)
        choices = [i.numpy() for i in choices]

        frequency = [0, 0, 0, 0]

        count = 0

        index = 0
        for i, env in enumerate(envs):
            pos = positions[i]
            len_pos = len(pos)
            cho = choices[index : (index + len_pos)]
            index += len_pos

            m = {}
            for i in range(len_pos):
                m[str(pos[i])] = cho[i]
                frequency[cho[i]] += 1

            stack = [(env.goal_x, env.goal_y)]

            def is_position(pos):
                x, y = pos
                return x >= 0 and x < env.width and y >= 0 and y < env.height

            len_m = len(m)

            while len(stack) > 0:
                x, y = stack.pop()
                count += 1 / (len_m * k)

                UP = (x, y + 1)
                if is_position(UP) and str(UP) in m and m[str(UP)] == Environment.UP:
                    stack.append(UP)

                DOWN = (x, y - 1)
                if (
                    is_position(DOWN)
                    and str(DOWN) in m
                    and m[str(DOWN)] == Environment.DOWN
                ):
                    stack.append(DOWN)

                RIGHT = (x - 1, y)
                if (
                    is_position(RIGHT)
                    and str(RIGHT) in m
                    and m[str(RIGHT)] == Environment.RIGHT
                ):
                    stack.append(RIGHT)

                LEFT = (x + 1, y)
                if (
                    is_position(LEFT)
                    and str(LEFT) in m
                    and m[str(LEFT)] == Environment.LEFT
                ):
                    stack.append(LEFT)

        return count, [i / k for i in frequency]

    def nan_check(self):
        for w, b in self.network:
            if has_invalid_numbers(w) or has_invalid_numbers(b):
                return True
        for w, b in self.target:
            if has_invalid_numbers(w) or has_invalid_numbers(b):
                return True
        return False


class ExpAgent(Agent):

    def create_environment(self) -> Environment:
        env = self.new_environment()
        x, y = env.maze.start
        env.set_position(x, y)

        for _ in range(
            (self.iter // self.target_update_frequency * self.updates_per_step) + 1
        ):
            env.move(self.random.randint(0, 3))

        return env
