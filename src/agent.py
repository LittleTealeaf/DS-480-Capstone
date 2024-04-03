from environment import Environment
import tensorflow as tf
from random import Random
from keras.activations import sigmoid
from keras.optimizers import SGD


def invalid_num_check(tensor):
    has_inf = tf.math.is_inf(tensor)
    has_nan = tf.math.is_nan(tensor)
    return tf.reduce_any(tf.logical_or(has_inf, has_nan))


@tf.function
def feed_forward(inputs, layers):
    variables = inputs
    for w, b in layers:
        variables = variables @ w
        variables = variables + b
        variables = sigmoid(variables)
    return variables


@tf.function
def feed_forward_argmax(inputs, layers):
    outputs = feed_forward(inputs, layers)
    return tf.argmax(outputs, 1)


@tf.function
def train(s_1, c_1, r_2, s_2, network, target, gamma, optimizer):
    variables = []

    with tf.GradientTape() as tape:
        for w, b in network:
            variables.extend([w, b])

        network_out = feed_forward(s_1, network)
        network_var_out = tf.gather(network_out, c_1, batch_dims=1)

        target_out = feed_forward(s_2, target)
        target_out_max = tf.reduce_max(target_out, axis=1)
        target_out_scaled = tf.multiply(gamma, target_out_max)

        loss_raw = target_out_scaled - network_var_out - r_2

        loss = tf.math.square(loss_raw)
        loss_mean = tf.reduce_mean(loss)

        gradient = tape.gradient(loss_mean, variables)
        optimizer.apply_gradient(zip(gradient, variables))


class Agent:
    """
    The agent used to train

    """

    def default_learning_rate(agent):
        return 0.1 * 0.9 ** (agent.epoch % agent.params.target_update_interval)

    def default_gamma(agent):
        return 0.8

    def default_epsilon(agent):
        return (
            0.5
            * (0.9 ** (agent.epoch // agent.params.target_update_interval))
            * (0.9 ** (agent.epoch % agent.params.target_update_frequency))
            + 0.3
        )

    def __init__(
        self,
        layer_sizes: list[int] = None,
        max_replay=10_000,
        height=10,
        width=10,
        seed=None,
        target_update_interval=100,
        step_update_interval=1,
        learning_rate=None,
        gamma=None,
        epsilon=None,
        training_seed=None,
        create_training_seed=False,
        evaluation_seed=None,
        create_evaluation_seed=False,
        evaluate_on_training=False,
    ):
        assert layer_sizes is not None

        if seed is None:
            seed = Random().random()

        if learning_rate is None:
            learning_rate = Agent.default_learning_rate

        if gamma is None:
            gamma = Agent.default_gamma

        if epsilon is None:
            epsilon = Agent.default_epsilon

        self.random = Random(seed)

        if create_training_seed and training_seed is None:
            training_seed = self.random.random()
            if evaluate_on_training:
                evaluation_seed = training_seed

        if create_evaluation_seed and evaluation_seed is None:
            evaluation_seed = self.random.random()

        self.params = {
            "layer_sizes": layer_sizes,
            "max_replay": max_replay,
            "height": height,
            "width": width,
            "seed": seed,
            "target_update_interval": target_update_interval,
            "step_update_interval": step_update_interval,
            "learning_rate": learning_rate,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "gamma": gamma,
            "epsilon": epsilon,
        }

        self.network = []

        layers = [i for i in layer_sizes]
        layers.append(4)

        prev = Environment(height, width).get_obs_length()
        tf.random.set_seed(self.random.random())
        for layer in layers:
            weights = tf.Variable(tf.random.normal((prev, layer)))
            bias = tf.Variable(tf.random.normal((layer,)))
            self.network.append((weights, bias))
            prev = layer
        self.update_target()

    def update_target(self):
        self.target = [
            (tf.constant(weights.numpy()), tf.constant(bias.numpy()))
            for weights, bias in self.network
        ]

    def create_environment(self, seed=None) -> Environment:
        return Environment(self.params.width, self.params.height, seed=seed)

    def build_environment(self, seed=None) -> Environment:
        return self.create_environment(seed=seed)

    def populate_replay(self, count: int):
        random = self.random

        if self.params.training_seed is not None:
            random = Random(self.params.training_seed)

        env = self.create_environment(seed=Environment.random_seed(random))
        for _ in range(count):
            while env.is_solved():
                env = self.create_environment(seed=Environment.random_seed(random))

            obs = env.get_observations()
            sel = self.random.randint(0, 3)



agent = Agent(layer_sizes=[1, 2, 3])
