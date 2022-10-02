# Author: Mattia Silvestri

"""
    Main methods to train and test the methods.
"""
from copy import deepcopy

import gym.vector
from pyagents.utils import get_optimizer

from agent import SACSE, SacMod, EditorNetwork, ExtendedQNetwork
from vpp_envs import SingleStepVPPEnv, MarkovianVPPEnv, SingleStepFullRLVPP, MarkovianRlVPPEnv
import numpy as np
import pandas as pd
import cloudpickle
import os
import argparse
import gin
import tensorflow as tf
from typing import Union, List
from pyagents import networks
from utility import timestamps_headers, make_env, METHODS, train_loop, test_agent

########################################################################################################################


MODES = ['train', 'test']
TIMESTEP_IN_A_DAY = 96


########################################################################################################################

@gin.configurable
def train_rl_algo(method: str = None,
                  algo: str = 'SAC',
                  safety_layer: bool = False,
                  step_reward: bool = False,
                  instances: Union[float, List[int]] = 0.25,
                  num_epochs: int = 1000,
                  noise_std_dev: Union[float, int] = 0.01,
                  batch_size: int = 100,
                  schedule: bool = False,
                  crit_learning_rate: float = 7e-4,
                  act_learning_rate: float = 7e-4,
                  alpha_learning_rate: float = 7e-4,
                  lambda_learning_rate: float = 0.01,
                  rollout_steps: int = 1,
                  train_steps: int = 1,
                  log_dir: str = None,
                  wandb_params: dict = None,
                  store_unfeasible: bool = False):
    """
    Training routing.
    :param method: string; choose among one of the available methods.
    :param instances: float or list of int; fraction or indexes of the instances to be used for test.
    :param num_epochs: int; number of training epochs.
    :param noise_std_dev: float; standard deviation for the additive gaussian noise.
    :param batch_size: int; batch size.
    :return:
    """

    env = make_env(method, instances, noise_std_dev, safety_layer=safety_layer, step_reward=step_reward)
    test_env = make_env(method, instances, noise_std_dev, safety_layer=True, step_reward=step_reward)
    # Set episode length and discount factor for single-step and MDP version
    if 'mdp' in method:
        max_episode_length = TIMESTEP_IN_A_DAY
        discount = 1.0
    elif 'single-step' in method:
        max_episode_length = 1
        discount = 0.0
    else:
        raise Exception("Method name must contain 'mdp' or 'single-step'")

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    bounds = (-1.0, 1.0)

    a_net = networks.PolicyNetwork(state_shape, action_shape,
                                   output='gaussian', bounds=bounds, activation='relu',
                                   out_params={'state_dependent_std': True,
                                               'mean_activation': None})
    if algo == 'SAC':
        q1_net = networks.QNetwork(state_shape=state_shape, action_shape=action_shape)
        q2_net = networks.QNetwork(state_shape=state_shape, action_shape=action_shape)

        log_dict = dict(act_learning_rate=act_learning_rate,
                        crit_learning_rate=crit_learning_rate,
                        alpha_learning_rate=alpha_learning_rate,
                        num_epochs=num_epochs, batch_size=batch_size,
                        schedule=schedule, store_unfeasible=store_unfeasible,
                        rollout_steps=rollout_steps, train_steps=train_steps)

        if schedule:
            act_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(act_learning_rate,
                                                                              num_epochs,
                                                                              0.)
            crit_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(crit_learning_rate,
                                                                               num_epochs,
                                                                               0.)
            alpha_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(alpha_learning_rate,
                                                                                num_epochs,
                                                                                0.)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c1_opt = get_optimizer(learning_rate=crit_learning_rate)
        c2_opt = get_optimizer(learning_rate=crit_learning_rate)
        alpha_opt = get_optimizer(learning_rate=alpha_learning_rate)

        agent = SacMod(state_shape, action_shape, buffer='uniform', gamma=discount,
                       actor=a_net, critic=q1_net, critic2=q2_net, reward_normalization=False,
                       actor_opt=a_opt, critic1_opt=c1_opt, critic2_opt=c2_opt, alpha_opt=alpha_opt,
                       target_update_period=1, reward_scaling=1.0,
                       wandb_params=wandb_params, save_dir=log_dir, log_dict=log_dict)

    elif algo == 'SACSE':
        editor_net = EditorNetwork(state_shape, action_shape,
                                   output='gaussian', bounds=bounds, activation='relu',
                                   out_params={'state_dependent_std': True,
                                               'mean_activation': None})
        reward_shape = (2,)
        q_nets = ExtendedQNetwork(state_shape, action_shape, reward_shape, n_critics=2)

        log_dict = dict(act_learning_rate=act_learning_rate,
                        crit_learning_rate=crit_learning_rate,
                        alpha_learøning_rate=alpha_learning_rate,
                        num_epochs=num_epochs, batch_size=batch_size,
                        schedule=schedule, store_unfeasible=store_unfeasible,
                        rollout_steps=rollout_steps, train_steps=train_steps)

        if schedule:
            act_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(act_learning_rate,
                                                                              num_epochs,
                                                                              0.)
            crit_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(crit_learning_rate,
                                                                               num_epochs,
                                                                               0.)
            alpha_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(alpha_learning_rate,
                                                                                num_epochs,
                                                                                0.)
            lambda_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(lambda_learning_rate,
                                                                                 num_epochs,
                                                                                 0.)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        ed_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        alpha_opt = get_optimizer(learning_rate=alpha_learning_rate)
        lambda_opt = get_optimizer(learning_rate=lambda_learning_rate)

        agent = SACSE(state_shape, action_shape, buffer='uniform', gamma=discount,
                      actor=a_net, editor=editor_net, critics=q_nets,
                      actor_opt=a_opt, editor_opt=ed_opt, critic_opt=c_opt, alpha_opt=alpha_opt,
                      lambda_opt=lambda_opt, train_lambda=True, target_update_period=1, reward_scaling=1.0,
                      wandb_params=wandb_params, save_dir=log_dir, log_dict=log_dict)
    else:
        raise Exception("either algo SAC or SACSE")
    agent.init(envs=env, min_memories=2000)
    agent = train_loop(agent, env, num_epochs, batch_size, rollout_steps, train_steps,
                       store_unfeasible=store_unfeasible, test_env=test_env)
    test_agent(agent, test_env)
    agent.save('_final')


########################################################################################################################


def test_rl_algo(log_dir: str,
                 predictions_filepath: str,
                 shifts_filepath: str,
                 prices_filepath: str,
                 method: str,
                 test_split: Union[float, List[int]],
                 num_episodes: int = 100):
    """
    Test a trained agent.
    :param log_dir: string; path where training information are saved to.
    :param predictions_filepath: string; where instances are loaded from.
    :param shifts_filepath: string; where optimal shifts are loaded from.
    :param prices_filepath: string; where prices are loaded from.
    :param method: string; choose among one of the available methods.
    :param test_split: float or list of int; fraction of the instances or list of instances indexes.
    :param num_episodes: int; number of episodes.
    :return:
    """
    # TODO use pyagents
    # Load parameters
    data = cloudpickle.load(open(os.path.join(log_dir, 'params.pkl'), 'rb'))
    # Get the agent
    algo = data['algo']
    env = data['env']

    # Load data from file
    # Check that all the required files exist
    assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
    assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
    assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"
    predictions = pd.read_csv(predictions_filepath)
    shift = np.load(shifts_filepath)
    c_grid = np.load(prices_filepath)

    # Split between training and test
    if isinstance(test_split, float):
        split_index = int(len(predictions) * (1 - test_split))
        train_predictions = predictions[:split_index]
    elif isinstance(test_split, list):
        split_index = test_split
        train_predictions = predictions.iloc[split_index]
    else:
        raise Exception("instances must be list of int or float")

    # Set episode length and discount factor for single-step and MDP version
    if 'mdp' in method:
        max_episode_length = TIMESTEP_IN_A_DAY
    elif 'single-step' in method:
        max_episode_length = 1
    else:
        raise Exception("Method name must contain 'mdp' or 'single-step'")

    if method == 'hybrid-mdp':
        # Create the environment
        env = MarkovianVPPEnv(predictions=train_predictions,
                              shift=shift,
                              c_grid=c_grid,
                              noise_std_dev=0,
                              savepath=None)

        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=max_episode_length)
        env = NormalizedEnv(env, normalize_obs=True)
    elif method == 'hybrid-single-step':
        # Create the environment
        env = SingleStepVPPEnv(predictions=train_predictions,
                               shift=shift,
                               c_grid=c_grid,
                               noise_std_dev=0,
                               savepath=None)

        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=max_episode_length)
    elif method == 'rl-single-step':
        # Create the environment
        env = SingleStepFullRLVPP(predictions=train_predictions,
                                  shift=shift,
                                  c_grid=c_grid,
                                  noise_std_dev=0,
                                  savepath=None)
        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=max_episode_length)
    elif method == 'rl-mdp':
        # Create the environment
        env = MarkovianRlVPPEnv(predictions=train_predictions,
                                shift=shift,
                                c_grid=c_grid,
                                noise_std_dev=0,
                                savepath=None)

        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=max_episode_length)

    # Get the policy
    policy = algo.policy

    timestamps = timestamps_headers(env.n)
    all_rewards = []

    total_reward = 0

    # Loop for each episode
    for episode in range(num_episodes):
        last_obs, _ = env.reset()
        done = False

        episode_reward = 0

        all_actions = []

        # Perform an episode
        while not done:
            # env.render(mode='ascii')
            _, agent_info = policy.get_action(last_obs)
            a = agent_info['mean']
            step = env.step(a)
            if 'action' in step.env_info:
                a = step.env_info['action']
            all_actions.append(np.squeeze(a))

            total_reward -= step.reward
            episode_reward -= step.reward

            if step.terminal or step.timeout:
                break
            last_obs = step.observation

        print(f'\nTotal reward: {episode_reward}')
        all_rewards.append(episode_reward)

        if method == 'rl-mdp':
            all_actions = np.expand_dims(all_actions, axis=0)
        elif method == 'rl-single-step':
            action = all_actions[0]
            storage_in = action[:env.n]
            storage_out = action[env.n:env.n * 2]
            grid_in = action[env.n * 2:env.n * 3]
            diesel_power = action[env.n * 3:]
            all_actions = np.stack([storage_in, storage_out, grid_in, diesel_power], axis=-1)

    if 'rl' in method:
        action_save_name = 'solution'
    else:
        action_save_name = 'cvirt'

    # Save the agent's actions
    all_actions = np.squeeze(all_actions)
    np.save(os.path.join(log_dir, action_save_name), all_actions)
    # TODO integrate test with plotting and wandb log (see experimental branch)


########################################################################################################################


if __name__ == '__main__':

    # NOTE: you should set the logging directory and the method
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Logging directory")
    parser.add_argument("--method", type=str, choices=METHODS,
                        help="'hybrid-single-step': this is referred to as 'single-step' in the paper;"
                             + "'hybrid-mdp': this is referred to as 'mdp' in the paper;"
                             + "'rl-single-step': end-to-end RL approach which directly provides the decision "
                             + "variables for all the stages;"
                             + "'rl-mdp': this is referred to as 'rl' in the paper.")
    parser.add_argument('-sl', '--safety-layer', action='store_true',
                        help="If True, use safety layer to correct unfeasible actions.")
    parser.add_argument('--step-reward', action='store_true',
                        help="If True, env returns step-by-step costs rather than cumulative cost at end of episode.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--n-instances", type=int, default=1, help="Number of instances the agent is trained on")
    parser.add_argument("--mode", type=str, choices=MODES, required=True,
                        help="'train': if you want to train a model from scratch;"
                             + "'test': if you want to test an existing model.")
    parser.add_argument('--gin', default=None, help='(Optional) path to .gin config file.')
    args = parser.parse_args()

    LOG_DIR = args.logdir
    METHOD = args.method

    mode = args.mode
    n_instances = args.n_instances

    safety_layer = args.safety_layer
    step_reward = args.step_reward

    if mode == 'train':
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size

        # Randomly choose n instances
        np.random.seed(0)
        indexes = np.arange(10000, dtype=np.int32)
        indexes = list(np.random.choice(indexes, size=n_instances))
    else:
        indexes = [int(x) for x in os.listdir(LOG_DIR)]

    print(indexes)
    if 'wandb.key' in os.listdir():
        key = (f := open('wandb.key')).read()
        f.close()
        tags = ['safety_layer'] if safety_layer else []
        tags += ['step_reward'] if step_reward else []
        tags += ['sac'] + list(map(lambda n: str(n), indexes))
        wandb_params = {'key': key,
                        'project': 'rl-online-offline-opt',
                        'entity': 'mazelions',
                        'tags': tags,
                        'group': '-'.join(map(lambda n: str(n), indexes))}
    else:
        wandb_params = None
    if args.gin is not None:
        gin.parse_config_file(args.gin)
    if mode == 'train':
        # Training routing
        train_rl_algo(method=METHOD,
                      safety_layer=safety_layer,
                      step_reward=step_reward,
                      instances=indexes,
                      num_epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      noise_std_dev=0.01,
                      wandb_params=wandb_params,
                      log_dir=LOG_DIR)
    elif mode == 'test':
        # Test trained methods
        for idx in indexes:
            test_rl_algo(log_dir=os.path.join(LOG_DIR, f'{idx}'),
                         predictions_filepath=os.path.join('data', 'Dataset10k.csv'),
                         shifts_filepath=os.path.join('data', 'optShift.npy'),
                         prices_filepath=os.path.join('data', 'gmePrices.npy'),
                         method=METHOD,
                         test_split=[idx],
                         num_episodes=1)
    else:
        raise Exception(f"{mode} is not supported")
