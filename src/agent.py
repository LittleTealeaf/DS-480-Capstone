from environment import Environment
import tensorflow as tf
from random import Random
from keras.activations import sigmoid
from keras.optimizers import SGD
from keras.optimizers.schedules import LearningRateSchedule as KerasLearningrateSchedule
from keras import backend


def invalid_num_check(tensor):
    has_inf = tf.math.is_inf(tensor)
    has_nan = tf.math.is_nan(tensor)
    return tf.reduce_any(tf.logical_or(has_inf, has_nan))


def feed_forward(inputs, layers):
    variables = inputs
    for w, b in layers:
        variables = variables @ w
        variables = variables + b
        variables = sigmoid(variables)
    return variables


def feed_forward_argmax(inputs, layers):
    outputs = feed_forward(inputs, layers)
    return tf.argmax(outputs, 1)


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
        optimizer.apply_gradients(zip(gradient, variables))


class Agent:
    """
    The agent used to train

    """

    def default_gamma(agent):
        return 0.8

    def default_epsilon(agent):
        return (
            0.3 * (0.99 ** agent.epoch)
            + 0.3
        )

    def default_learning_rate(agent):
        return (
            0.01 * (
            0.99 ** (agent.epoch % agent.params["target_update_interval"]))
            # (0.99 ** (agent.epoch // agent.params["target_update_interval"]))
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
        gamma=None,
        epsilon=None,
        use_single_maze = False,
        training_seed=None,
        create_training_seed=False,
        evaluation_seed=None,
        create_evaluation_seed=False,
        evaluate_on_training=False,
        learning_rate=None,
    ):
        assert layer_sizes is not None

        if seed is None:
            seed = Random().random()

        if gamma is None:
            gamma = Agent.default_gamma

        if epsilon is None:
            epsilon = Agent.default_epsilon

        self.random = Random(seed)
        self.replay = []
        self.epoch = 0

        self.learning_rate = tf.Variable(0.1,trainable=False)

        if use_single_maze:
            self.maze_seed = Environment.random_seed(self.random)
        else:
            self.maze_seed = None

        if create_training_seed and training_seed is None:
            training_seed = self.random.random()
            if evaluate_on_training:
                evaluation_seed = training_seed

        if create_evaluation_seed and evaluation_seed is None:
            evaluation_seed = self.random.random()

        if learning_rate is None:
            learning_rate = Agent.default_learning_rate


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
            "use_single_maze": use_single_maze
        }


        self.optimizer = SGD(learning_rate=self.learning_rate)


        self.tf_train = tf.function(train)
        self.tf_feed_forward = tf.function(feed_forward)
        self.tf_feed_forward_argmax = tf.function(feed_forward_argmax)

        self.network = []

        self.update_target()
        self.update_target()

    def update_target(self):
        self.target = [
            (tf.constant(weights.numpy()), tf.constant(bias.numpy()))
            for weights, bias in self.network
        ]
        self.network = []

        # update network to squishimifcation

        layers = [i for i in self.params["layer_sizes"]]
        layers.append(4)
        height, width = self.params["height"], self.params["width"]


        prev = Environment(height, width).get_obs_length()
        tf.random.set_seed(self.random.random())
        for layer in layers:
            weights = tf.Variable(tf.random.normal((prev, layer)))
            bias = tf.Variable(tf.random.normal((layer,)))
            self.network.append((weights, bias))
            prev = layer


    def create_environment(self, seed=None) -> Environment:
        if self.maze_seed is None:
            return Environment(self.params["width"], self.params["height"], seed=seed)
        else:
            return Environment(self.params["width"], self.params["height"],seed=self.maze_seed)

    def build_environment(self, seed=None) -> Environment:
        return self.create_environment(seed=seed)

    def populate_replay(self, count: int):
        random = self.random

        if self.params["training_seed"] is not None:
            random = Random(self.params["training_seed"])

        env = self.create_environment(seed=Environment.random_seed(random))
        for _ in range(count):
            while env.is_solved():
                seed = Environment.random_seed(random)
                env = self.create_environment(seed=seed)


            obs = env.get_observations()
            sel = self.random.randint(0, 3)
            if self.params["epsilon"](self) < self.random.random():
                shape = (1, obs.size)
                obs_tf = tf.constant(obs, dtype=tf.float32, shape=shape)
                sel = int(self.tf_feed_forward_argmax(obs_tf, self.network))

            if env.move(sel):
                reward = env.get_reward()
            else:
                reward = -1.0
            obs_2 = env.get_observations()

            if len(self.replay) > self.params["max_replay"]:
                index = self.random.randint(0, self.params["max_replay"])
                self.replay[index] = (obs, sel, obs_2, reward)
            else:
                self.replay.append((obs, sel, obs_2, reward))

    def train(self, count: int):
        self.epoch += 1

        values = self.random.choices(self.replay, k=count)
        state_1 = []
        choices = []
        rewards = []
        state_2 = []
        for st1, ch, st2, rew in values:
            state_1.append(st1)
            state_2.append(st2)
            choices.append(ch)
            rewards.append(rew)

        state_1_tf = tf.constant(state_1, dtype=tf.float32)
        state_2_tf = tf.constant(state_2, dtype=tf.float32)
        choices_tf = tf.constant(choices)
        rewards_tf = tf.constant(rewards, dtype=tf.float32)

        backend.set_value(self.learning_rate, self.params["learning_rate"](self))

        self.tf_train(
            state_1_tf,
            choices_tf,
            rewards_tf,
            state_2_tf,
            self.network,
            self.target,
            self.params["gamma"](self),
            self.optimizer,
        )

        if self.epoch % self.params["target_update_interval"] == 0:
            self.update_target()

    def evaluate(self, count: int = 1):
        random = self.random

        if self.params["evaluation_seed"] is not None:
            random = Random(self.params["evaluation_seed"])

        envs = [
            self.build_environment(Environment.random_seed(random))
            for _ in range(count)
        ]

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

        choices = self.tf_feed_forward_argmax(observations_tf, self.network)
        choices = [i.numpy() for i in choices]

        frequency = [0, 0, 0, 0]
        counts = 0
        index = 0

        for i, env in enumerate(envs):
            pos = positions[i]
            len_pos = len(pos)
            last_index = index + len_pos
            cho = choices[index:last_index]
            index += len_pos

            m = {}
            for i in range(len_pos):
                m[str(pos[i])] = cho[i]
                frequency[cho[i]] += 1

            stack = [(env.goal_x, env.goal_y)]

            len_m = len(m)

            def is_valid_position(pos):
                return env.is_valid_position(pos[0], pos[1])

            counts_sub = -1

            while len(stack) > 0:
                x, y = stack.pop()
                counts_sub += 1

                UP = (x, y + 1)
                if (
                    is_valid_position(UP)
                    and str(UP) in m
                    and m[str(UP)] == Environment.UP
                ):
                    stack.append(UP)

                DOWN = (x, y - 1)
                if (
                    is_valid_position(DOWN)
                    and str(DOWN) in m
                    and m[str(DOWN)] == Environment.DOWN
                ):
                    stack.append(DOWN)

                RIGHT = (x - 1, y)
                if (
                    is_valid_position(RIGHT)
                    and str(RIGHT) in m
                    and m[str(RIGHT)] == Environment.RIGHT
                ):
                    stack.append(RIGHT)

                LEFT = (x + 1, y)
                if (
                    is_valid_position(LEFT)
                    and str(LEFT) in m
                    and m[str(LEFT)] == Environment.LEFT
                ):
                    stack.append(LEFT)
            counts += counts_sub / (count * (len_m - 1))
        return counts, [i / count for i in frequency]

    def print_policy(self, maze = None):
        if maze is None:
            maze = self.create_environment()
        ls = [[' ' if j == 0 else ' ' for j in i] for i in maze.maze.grid]

        observations = []
        valids = maze.get_valid_positions()
        obs_len = maze.get_obs_length()

        for x,y in valids:
            assert maze.set_position(x,y)
            observations.append(maze.get_observations())

        observations_tf = tf.constant(
            observations, shape=(len(observations), obs_len), dtype=tf.float32
        )

        choices = self.tf_feed_forward_argmax(observations_tf, self.network)
        choices = [i.numpy() for i in choices]

        for i in range(len(valids)):
            x,y = valids[i]
            if choices[i] == Environment.UP:
                ls[y][x] = '^'
            elif choices[i] == Environment.DOWN:
                ls[y][x] = 'v'
            elif choices[i] == Environment.RIGHT:
                ls[y][x] = '>'
            else:
                ls[y][x] = '<'

        ls[maze.goal_y][maze.goal_x] = 'G'

        print("\n".join(["".join([j for j in i]) for i in ls]))

    def has_nan_inf(self):
        for w, b in self.network:
            if invalid_num_check(w) or invalid_num_check(b):
                return True
        return False

    def to_exp_agent(self):
        return ExpAgent(**self.params)

    def to_normal_agent(self):
        return Agent(**self.params)


class ExpAgent(Agent):
    def build_environment(self, seed=None) -> Environment:
        env = self.create_environment(seed)
        x, y = env.maze.start
        env.set_position(x, y)

        for _ in range(
            (
                self.epoch
                // (self.params["target_update_interval"]
                * self.params["step_update_interval"])
            )
            + 1
        ):
            move = self.random.choice([
                i for i in [Environment.UP, Environment.DOWN, Environment.LEFT, Environment.RIGHT] if env.can_move(i)
            ])
            env.move(move)
            # env.move(self.random.randint(0, 3))
        return env
