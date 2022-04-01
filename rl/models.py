# Author Mattia Silvestri

"""
    Tensorflow 2 models for RL algorithms.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from rl.baselines import Critic
from rl.utility import from_dict_of_tensor_to_numpy

########################################################################################################################

DISCRETE_SPACE = "discrete"
CONTINUOUS_SPACE = "continuous"


########################################################################################################################

class DistributionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DistributionLayer, self).__init__(**kwargs)

    def call(self, x):
        raise NotImplementedError()

    def log_probs(self, output, actions):
        raise NotImplementedError()


class GaussianLayer(DistributionLayer):
    def __init__(self, output_dim, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self._actor_mean = Dense(output_dim)
        self._actor_std_dev = Dense(output_dim, activation=tf.math.softplus)

    def call(self, x):
        mean = self._actor_mean(x)
        std_dev = self._actor_std_dev(x)
        return mean, std_dev

    def log_probs(self, output, actions):
        gaussian = tfp.distributions.MultivariateNormalDiag(loc=output[0], scale_diag=output[1])
        return gaussian.log_prob(actions)


class DirichletLayer(DistributionLayer):
    def __init__(self, output_dim, **kwargs):
        super(DirichletLayer, self).__init__(**kwargs)
        self._actor_alpha = Dense(output_dim, activation=tf.math.softplus, bias_initializer=tf.keras.initializers.Ones())

    def call(self, x):
        alpha = self._actor_alpha(x)
        eps = tf.where(alpha < 0.3, 0.3, 0)
        return alpha + eps

    def log_probs(self, output, actions):
        dirichlet = tfp.distributions.Dirichlet(output)
        return dirichlet.log_prob(actions)


########################################################################################################################


class DRLModel(tf.keras.Model):
    """
    Deep Reinforcement Learning base class.
    """

    def __init__(self, input_shape, output_dim, hidden_units=(64, 64)):
        """
        :param output_dim: int; output dimension of the neural network, i.e. the actions space.
        :param hidden_units: list of int; units for each hidden layer.
        """

        super(DRLModel, self).__init__()
        self._output_dim = output_dim

        # Define common body
        self._model = Sequential()
        self._model.add(Input(input_shape))
        self._model.add(Dense(units=hidden_units[0], activation='tanh'))
        for units in hidden_units[1:]:
            self._model.add(Dense(units=units, activation='tanh'))

        # Create the actor
        self.actor_output = DirichletLayer(output_dim)

        # Call build method to define the input shape
        self.build((None,) + input_shape)
        self.compute_output_shape(input_shape=(None,) + input_shape)

        # Define optimizer
        self._policy_optimizer = Adam(learning_rate=3e-4)

        # Keep track of trainable variables
        self._actor_trainable_vars = list()
        self._actor_trainable_vars += self._model.trainable_variables
        self._actor_trainable_vars += self.actor_output.trainable_variables

    def call(self, inputs):
        """
        Override the call method of tf.keras Model.
        :param inputs: numpy.array or tf.Tensor; the input arrays.
        :return: tf.Tensor; the output logits.
        """
        hidden_state = self._model(inputs)
        return self.actor_output(hidden_state)

    @from_dict_of_tensor_to_numpy
    @tf.function
    def train_step(self, *args, **kwargs):
        """
        A single training step.
        """
        raise NotImplementedError()

    @property
    def actor_trainable_vars(self):
        return self._actor_trainable_vars

    def apply_gradients(self, gradient):
        self._policy_optimizer.apply_gradients(zip(gradient, self._actor_trainable_vars))

########################################################################################################################


class PolicyGradient(DRLModel):
    """
        Definition of Policy Gradient RL algorithm.
    """

    def __init__(self, input_shape, output_dim, hidden_units=(32, 32)):
        super(PolicyGradient, self).__init__(input_shape, output_dim, hidden_units)

    @from_dict_of_tensor_to_numpy
    # @tf.function
    def train_step(self, states, q_vals, adv, actions):
        """
        Compute loss and gradients. Perform a training step.
        :param states: numpy.array; states of sampled trajectories.
        :param q_vals: list of float; expected return computed with Monte Carlo sampling.
        :param adv: numpy.array; the advantage for each action in the sampled trajectories.
        :param actions: numpy.array; actions of sampled trajectories.
        :return: loss: float; policy loss value.
        """

        # Tape the gradient during forward step and loss computation
        with tf.GradientTape() as policy_tape:
            output = self.call(inputs=states)
            log_prob = self.actor_output.log_probs(output, actions)
            policy_loss = tf.reduce_mean(tf.multiply(-log_prob, adv))
        assert not tf.math.is_inf(policy_loss)

        # Perform un update step
        for watched_var, trained_var in zip(policy_tape.watched_variables(), self._actor_trainable_vars):
            assert watched_var.ref() == trained_var.ref()
        dloss_policy = policy_tape.gradient(policy_loss, self._actor_trainable_vars)
        self._policy_optimizer.apply_gradients(zip(dloss_policy, self._actor_trainable_vars))

        return {'Policy loss': policy_loss}


########################################################################################################################


class A2C(PolicyGradient):
    """
        Definition of Advantage Actor-Critic RL algorithm.
    """

    def __init__(self, input_shape, output_dim, critic: Critic, hidden_units=(32, 32)):
        super(A2C, self).__init__(input_shape, output_dim, hidden_units)

        self.critic = critic

    @from_dict_of_tensor_to_numpy
    # @tf.function
    def train_step(self, states, q_vals, actions):
        """
        Compute loss and gradients. Perform a training step.
        :param states: numpy.array; states of sampled trajectories.
        :param q_vals: list of float; expected return computed with Monte Carlo sampling.
        :param actions: numpy.array; actions of sampled trajectories.
        :return: loss: float; policy loss value.
        """
        # FIXME verificare che adv sia tensore
        adv = self.critic.compute_advantage(states, q_vals)
        loss_info = super().train_step(states, q_vals, adv, actions)

        # Perform critic update step
        critic_loss = self.critic.train_step(states, tf.reshape(q_vals, shape=[-1, 1]))
        loss_info['Critic loss'] = critic_loss
        return loss_info


########################################################################################################################

class Editor(DRLModel):
    def __init__(self, input_shape, output_dim, critic: Critic, hidden_units=(32, 32)):
        super(Editor, self).__init__(input_shape, output_dim, hidden_units)
        self.critic = critic

    def call(self, inputs):
        if tf.nest.is_nested(inputs):
            assert isinstance(inputs, (tuple, list)), f'Needs to be a tuple or a list of tensors'
            inputs = tf.concat(inputs, axis=-1)
        return super().call(inputs)

    def train_step(self, states, q_vals, actions):
        adv = self.critic.compute_advantage(states, q_vals)
        # Tape the gradient during forward step and loss computation
        with tf.GradientTape() as policy_tape:
            output = self.call((states, actions))
            log_prob = self.actor_output.log_probs(output, actions)
            policy_loss = tf.reduce_mean(tf.multiply(-log_prob, adv))
        assert not tf.math.is_inf(policy_loss)

        # Perform un update step
        for watched_var, trained_var in zip(policy_tape.watched_variables(), self._actor_trainable_vars):
            assert watched_var.ref() == trained_var.ref()
        dloss_policy = policy_tape.gradient(policy_loss, self._actor_trainable_vars)
        self._policy_optimizer.apply_gradients(zip(dloss_policy, self._actor_trainable_vars))

        # Perform critic update step
        critic_loss = self.critic.train_step(states, tf.reshape(q_vals, shape=[-1, 1]))

        return {'Policy loss': policy_loss, 'Critic loss': critic_loss}