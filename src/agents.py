from environment import Environment
import tensorflow as tf
from random import Random
from keras.activations import sigmoid
from keras.optimizers import SGD


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
    def __init__(self, layer_sizes: list[int], max_replay=10_000, height=10, width=10):
        layer_sizes.append(4)
        self.network = []
        self.target = []
        self.replay = []
        self.iter = 0
        self.random = Random()
        self.max_replay = max_replay
        self.height = 10
        self.width = 10
        self.obs_size = Environment(height, width).get_obs_length()
        self.optimizer = None

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
        return 0.1

    def gamma(self):
        return 0.5

    def epsilon(self):
        return 0.2

    def create_environment(self) -> Environment:
        return Environment(self.height, self.width)

    def populate_replay(self, count: int):
        env = self.create_environment()
        for _ in range(count):
            if env.is_solved():
                env = self.create_environment()

            obs = env.get_observations()
            sel = self.random.randrange(0, 4)
            if self.epsilon() < self.random.random():
                obs_tf = tf.constant(obs, dtype=tf.float32, shape=(1, obs.size))
                sel = int(feed_forward_argmax(obs_tf, self.network))

            env.move(sel)
            obs_2 = env.get_observations()

            reward = env.get_reward()

            l = len(self.replay)
            while l >= self.max_replay:
                self.replay.remove(self.random.randint(0, l))
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


class ExpAgent(Agent):
    def create_environment(self) -> Environment:
        return Environment(self.height, self.width)
