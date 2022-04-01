# Author: Mattia Silvestri

"""
    RL agents.
"""

import numpy as np
import wandb

from rl.utility import calc_qvals
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


########################################################################################################################


class DRLAgent(tf.Module):
    """
    Abstract class for Deep Reinforcement Learning agent.
    """

    def __init__(self, env, policy, model, baseline, standardize_q_vals, wandb_log=False):
        """

        :param env: gym.Environment; the agent interacts with this environment.
        :param policy: policy.Policy; policy defined as a probability distribution of actions over states.
        :param model: model.DRLModel; DRL model.
        :param baseline: baselines.Baseline; baseline used to reduce the variance of the Q-values.
        """

        super().__init__(**kwargs)
        self._env = env
        self._policy = policy
        self._model = model
        self._baseline = baseline
        self._standardize_q_vals = standardize_q_vals
        self._wandb_log = wandb_log

    def act(self, state):
        probs = self._model(np.expand_dims(state, axis=0))
        action = self._policy.select_action(probs)
        return action, None

    def _step(self, action):
        """
        Private method to ensure environment input action is given in the proper format to Gym.
        :param action: numpy.array; the action.
        :return:
        """
        if isinstance(self._env.action_space, Discrete):
            assert action.shape == (self._env.action_space.n,)
            assert np.sum(action) == 1

            action = np.argmax(action)

            return self._env.step(action)
        elif isinstance(self._env.action_space, Box):
            assert action.shape == self._env.action_space.shape

            return self._env.step(action)

    def _log(self, prefix='', **kwargs):
        if self._wandb_log:
            wandb.log({f'{prefix}{"/" if prefix else ""}{k}': v for k, v in kwargs.items()})

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; number of interactions with the environment for training.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where to save/load model weights.
        :return:
        """

        raise NotImplementedError()

    def test(self, loadpath, render, num_episodes=1):
        """
        Test the model.
        :param loadpath: string; model weights loadpath.
        :param render: bool; True if you want to visualize the environment, False otherwise.
        :param num_episodes: int; the number of episodes.
        :return:
        """

        # Load model
        self._model = load_model(loadpath)
        train_with_safety_layer = self._env.safety_layer
        self._env.safety_layer = True

        # Loop over the number of episodes
        for _ in range(num_episodes):

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()
            score = 0
            all_actions = []

            # Perform an episode
            while not game_over:

                # Render if required
                if render:
                    self._env.render()

                # Sample an action from policy
                # Add the batch dimension for the NN model
                a_t, _ = self.act(s_t)

                # Perform a step
                s_tp1, rs_t, game_over, info = self._step(a_t)
                if 'action' in info:
                    a_t = info['action']
                all_actions.append(np.squeeze(a_t))
                s_t = s_tp1
                score += rs_t['cost']

            print('Score: {}'.format(score))
            self._log('test', score=score, constraints=rs_t['constraints'])

        all_actions = np.squeeze(all_actions)
        self._env.safety_layer = train_with_safety_layer
        return all_actions


########################################################################################################################


class OnPolicyAgent(DRLAgent):
    """
    DRL agent which requires on-policy samples.
    """

    def __init__(self, env, policy, model, baseline, standardize_q_vals, **kwargs):
        """

        :param env: environment on which to train the agent; as Gym environment
        :param policy: policy defined as a probability distribution of actions over states; as policy.Policy
        :param model: DRL model; as models.DRLModel
        :param baseline: baselines.Baseline; baseline used to reduce the variance of the Q-values.
        """

        super(OnPolicyAgent, self).__init__(env, policy, model, baseline, standardize_q_vals, **kwargs)

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; training steps in the environment.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where model weights will be saved.
        :return:
        """

        # Training steps
        steps = 0

        # Sampled trajectory variables
        actions = list()
        states = list()
        q_vals = list()

        score = 0
        num_episodes = 0

        while steps < num_steps:

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()

            # Reset current episode states, actions and rewards
            current_states = list()
            current_actions = list()
            current_rewards = list()

            # Keep track of the episode number
            num_episodes += 1

            # Perform an episode
            while not game_over:

                # Render the environment if required
                if render:
                    self._env.render()

                # Sample an action from policy
                # Add the batch dimension for the NN model
                action = self.act(s_t)
                current_actions.append(action)

                # Sample current state, next state and reward
                current_states.append(s_t)
                s_tp1, r_t, game_over, _ = self._step(action)
                current_rewards.append(r_t)
                s_t = s_tp1

                # Increase the score and the steps counter
                score += r_t
                steps += 1

            # Compute the Q-values
            current_q_vals = calc_qvals(current_rewards,
                                        gamma=gamma,
                                        max_episode_length=self._env.max_episode_length)

            # Keep track of trajectories
            states = states + current_states
            actions = actions + current_actions
            q_vals.append(current_q_vals)

            # Training step
            if len(states) >= batch_size:
                # Convert trajectories from list to array
                states = np.asarray(states)
                actions = np.asarray(actions)
                q_vals = np.asarray(q_vals)

                if self._standardize_q_vals:
                    mean = np.nanmean(q_vals, axis=0)
                    std = np.nanstd(q_vals, axis=0)
                    q_vals = (q_vals - mean) / (std + 1e-5)

                # Compute advatange
                adv = self._baseline.compute_advantage(states, q_vals)

                # Perform a gradient descent step
                # Convert states, Q-values and advantage to tensor
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                adv = tf.convert_to_tensor(adv, dtype=tf.float32)
                q_vals = tf.convert_to_tensor(q_vals[~np.isnan(q_vals)], dtype=tf.float32)
                loss_dict = self._model.train_step(states, q_vals, adv, actions)

                # Visualization and logging
                self._log('train', score=score, episodes=num_episodes, avg_score=score / num_episodes, **loss_dict)

                print_string = 'Frame: {}/{} | Total reward: {:.2f}'.format(steps, num_steps, score)
                print_string += ' | Total number of episodes: {} | Average score: {:.2f}'.format(num_episodes,
                                                                                                 score / num_episodes)
                for loss_name, loss_value in loss_dict.items():
                    print_string += ' | {}: {:.5f} '.format(loss_name, loss_value)

                print(print_string + '\n')
                print('-' * len(print_string) + '\n')

                # Clear trajectory variables
                states = list()
                actions = list()
                q_vals = list()

                # Reset score and number of episodes
                score = 0
                num_episodes = 0

        # Save model
        if filename is not None:
            self._model.save(filename)


########################################################################################################################
# Authors Chinellato Diego & Campardo Giorgia


class SafetyEditorAgent(DRLAgent):
    """
    DRL agent which requires on-policy samples.
    """

    def __init__(self, env, policy, model, baseline, editor, standardize_q_vals, **kwargs):
        """

        :param env: environment on which to train the agent; as Gym environment
        :param policy: policy defined as a probability distribution of actions over states; as policy.Policy
        :param model: DRL model; as models.DRLModel
        :param baseline: baselines.Baseline; baseline used to reduce the variance of the Q-values.
        """

        super(OnPolicySafetyEditorAgent, self).__init__(env, policy, model, baseline, standardize_q_vals, **kwargs)
        self._editor = editor
        self.lagrangian = lagrangian

        self._alpha_optim = Adam(learning_rate=1e-4)
        self.log_alpha = (tf.Variable(1., trainable=True), tf.Variable(1., trainable=True))
        self._target_entropy = (4., 4.)

    def act(self, state):
        probs = self._model(np.expand_dims(state, axis=0))
        hat_action = self._policy.select_action(probs)
        delta_probs = self._editor((np.expand_dims(state, axis=0), hat_action))
        delta_action = self._policy.select_action(delta_probs)
        action = self._safe_action(hat_action, delta_action)
        action = tf.squeeze(action)
        return action, (tf.squeeze(delta_action), tf.squeeze(hat_action))

    def _alpha_train_step(self, a_entropy, da_entropy):
        with tf.GradientTape() as alpha_tape:
            alpha_loss = tf.reduce_mean(self.log_alpha[0] * tf.stop_gradient(a_entropy - self._target_entropy[0]))
            alpha_loss += tf.reduce_mean(self.log_alpha[1] * tf.stop_gradient(da_entropy - self._target_entropy[1]))
        grads_alpha = alpha_tape.gradient(alpha_loss, self.log_alpha)
        self._alpha_optim.apply_gradients(zip(grads_alpha, self.log_alpha))
        return float(alpha_loss)

    def _train_step(self, states, qvals_cost, qvals_cons, actions, ahats, adeltas):
        adv_model = self._model.critic.compute_advantage(states, qvals_cost)
        adv_editor = self._editor.critic.compute_advantage(states, qvals_cons)

        with tf.GradientTape() as ahat_tape, tf.GradientTape() as da_tape:
            ahat_tape.watch(self._model.actor_trainable_vars)
            da_tape.watch(self._editor.actor_trainable_vars)
            alpha = tf.stop_gradient(tf.exp(self.log_alpha[0]))
            alpha_editor = tf.stop_gradient(tf.exp(self.log_alpha[1]))

            dist_params = self._model(states)
            # _ = self._policy.select_action(dist_params)
            delta_dist_params = self._editor((states, ahats))
            # _ = self._policy.select_action(delta_dist_params)
            # action = self._safe_action(hat_action, delta_action)

            log_probs_model = self._model.actor_output.log_probs(dist_params, ahats)
            loss_model = - tf.reduce_mean(tf.multiply(log_probs_model, adv_model) + alpha * log_probs_model)

            a_l2 = tf.math.reduce_sum(tf.math.square(actions - ahats), axis=-1)
            log_probs_editor = self._editor.actor_output.log_probs(delta_dist_params, adeltas)
            loss_editor = - tf.reduce_mean(a_l2 + tf.multiply(log_probs_editor, adv_editor) + alpha_editor * log_probs_editor)

        grads_ahat = ahat_tape.gradient(loss_model, self._model.actor_trainable_vars)
        self._model.apply_gradients(grads_ahat)
        grads_da = da_tape.gradient(loss_editor, self._editor.actor_trainable_vars)
        self._editor.apply_gradients(grads_da)

        model_critic_loss = self._model.critic.train_step(states, tf.reshape(qvals_cost, (-1, 1)))
        editor_critic_loss = self._editor.critic.train_step(states, tf.reshape(qvals_cons, (-1, 1)))

        alpha_loss = self._alpha_train_step(log_probs_model, log_probs_editor)

        return {'model_actor_loss': float(loss_model), 'editor_actor_loss': float(loss_editor), 'alpha_loss': alpha_loss,
                'model_critic_loss': float(model_critic_loss), 'editor_critic_loss': float(editor_critic_loss)}

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; training steps in the environment.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where model weights will be saved.
        :return:
        """

        # Training steps
        steps = 0

        # Sampled trajectory variables
        actions = list()
        ahats = list()
        adeltas = list()
        states = list()
        q_vals_cost = list()
        q_vals_constraints = list()

        score = 0
        num_episodes = 0

        while steps < num_steps:

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()

            # Reset current episode states, actions and rewards
            current_states = list()
            current_actions = list()
            current_ahats = list()
            current_adeltas = list()
            current_costs = list()
            current_constraints = list()

            # Keep track of the episode number
            num_episodes += 1

            # Perform an episode
            while not game_over:

                # Render the environment if required
                if render:
                    self._env.render()

                action, (hat_action, delta_action) = self.act(s_t)

                current_actions.append(action)
                current_ahats.append(hat_action)
                current_adeltas.append(delta_action)

                # Sample current state, next state and reward
                current_states.append(s_t)
                s_tp1, rs_t, game_over, _ = self._step(action)
                rcost_t = rs_t['cost']
                rcons_t = rs_t['constraints']
                current_costs.append(rcost_t)
                current_constraints.append(rcons_t)
                s_t = s_tp1

                # Increase the score and the steps counter
                score += rcost_t
                steps += 1

            # Compute the Q-values
            current_q_vals_cost = calc_qvals(current_costs,
                                             gamma=gamma,
                                             max_episode_length=self._env.max_episode_length)
            current_q_vals_cons = calc_qvals(current_constraints,
                                             gamma=1,
                                             max_episode_length=self._env.max_episode_length)
            current_q_vals_cons *= self.lagrangian

            # Keep track of trajectories
            states = states + current_states
            actions = actions + current_actions
            ahats = ahats + current_ahats
            adeltas = adeltas + current_adeltas
            q_vals_cost.append(current_q_vals_cost)
            q_vals_constraints.append(current_q_vals_cons)

            # Training step
            if len(states) >= batch_size:
                # Convert trajectories from list to array
                states = np.asarray(states)
                actions = np.asarray(actions)
                ahats = np.asarray(ahats)
                adeltas = np.asarray(adeltas)
                q_vals_cost = np.concatenate(q_vals_cost).reshape(-1, 1)
                q_vals_cons = np.concatenate(q_vals_constraints).reshape(-1, 1)

                if self._standardize_q_vals:
                    mean = np.nanmean(q_vals_cost, axis=0)
                    std = np.nanstd(q_vals_cost, axis=0)
                    q_vals_cost = (q_vals_cost - mean) / (std + 1e-5)

                    mean = np.nanmean(q_vals_cons, axis=0)
                    std = np.nanstd(q_vals_cons, axis=0)
                    q_vals_cons = (q_vals_cons - mean) / (std + 1e-5)

                # Perform a gradient descent step
                # Convert states, Q-values and advantage to tensor
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                adv = tf.convert_to_tensor(adv, dtype=tf.float32)
                q_vals = tf.convert_to_tensor(q_vals[~np.isnan(q_vals)], dtype=tf.float32)
                loss_dict = self._model.train_step(states, q_vals, adv, actions)
                # TODO train_step of the editor (be careful with tensors and detach)

                # Visualization and logging
                self._log('train', score=score, constraints=rcons_t, episodes=num_episodes,
                          avg_score=score / num_episodes, **loss_dict)

                print_string = 'Frame: {}/{} | Total reward: {:.2f}'.format(steps, num_steps, score)
                print_string += ' | Total number of episodes: {} | Average score: {:.2f}'.format(num_episodes,
                                                                                                 score / num_episodes)
                for loss_name, loss_value in loss_dict.items():
                    print_string += ' | {}: {:.5f} '.format(loss_name, loss_value)

                print(print_string + '\n')
                print('-' * len(print_string) + '\n')

                # Clear trajectory variables
                actions = list()
                ahats = list()
                adeltas = list()
                states = list()
                q_vals_cost = list()
                q_vals_constraints = list()

                # Reset score and number of episodes
                score = 0
                num_episodes = 0

        # Save model
        if filename is not None:
            self._model.save(filename)

########################################################################################################################
