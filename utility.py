# Author: Mattia Silvestri

"""
    Utility methods.
"""
import os
from datetime import datetime, timedelta

import gym
import numpy as np
import pandas as pd
from typing import Tuple, Union, List

import tqdm
import wandb

import vpp_envs
import online_heuristic

METHODS = ['hybrid-single-step', 'hybrid-mdp', 'rl-single-step', 'rl-mdp']


########################################################################################################################

def make_env(method, instances, noise_std_dev: Union[float, int] = 0.01,
             safety_layer: bool = False,
             step_reward: bool = False):
    # set_seed(1)
    # FIXME: the filepath should not be hardcoded
    predictions_filepath = os.path.join('data', 'Dataset10k.csv')
    prices_filepath = os.path.join('data', 'gmePrices.npy')
    shifts_filepath = os.path.join('data', 'optShift.npy')

    # Check that the selected method is valid
    assert method in METHODS, f"{method} is not valid"
    print(f'Selected method: {method}')

    # Load data from file
    # Check that all the required files exist
    assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
    assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
    assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"
    predictions = pd.read_csv(predictions_filepath)
    shift = np.load(shifts_filepath)
    c_grid = np.load(prices_filepath)

    # Split between training and test
    if isinstance(instances, float):
        split_index = int(len(predictions) * (1 - instances))
        train_predictions = predictions[:split_index]
    elif isinstance(instances, list):
        train_predictions = predictions.iloc[instances]
    else:
        raise Exception("test_split must be list of int or float")

    if method == 'hybrid-mdp':
        # Create the environment
        env = vpp_envs.MarkovianVPPEnv(predictions=train_predictions,
                                       shift=shift,
                                       c_grid=c_grid,
                                       noise_std_dev=noise_std_dev,
                                       savepath=None)
    elif method == 'hybrid-single-step':
        # Create the environment
        env = vpp_envs.SingleStepVPPEnv(predictions=train_predictions,
                                        shift=shift,
                                        c_grid=c_grid,
                                        noise_std_dev=noise_std_dev,
                                        savepath=None)
    elif method == 'rl-single-step':
        # Create the environment
        env = vpp_envs.SingleStepFullRLVPP(predictions=train_predictions,
                                           shift=shift,
                                           c_grid=c_grid,
                                           noise_std_dev=noise_std_dev,
                                           savepath=None)
    elif method == 'rl-mdp':
        # Create the environment
        env = vpp_envs.MarkovianRlVPPEnv(predictions=train_predictions,
                                         shift=shift,
                                         c_grid=c_grid,
                                         noise_std_dev=noise_std_dev,
                                         savepath=None,
                                         safety_layer=safety_layer,
                                         step_reward=step_reward)
    else:
        raise Exception(f'Method name must be in {METHODS}')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


########################################################################################################################


def train_loop(agent, env, num_epochs, batch_size, rollout_steps=1, train_steps=1, test_every=200, test_env=None):
    k = 1
    episode = 0
    # Test untrained agent
    best_score, sl_usage, l2_dists = test_agent(agent, test_env, render_plots=False)
    wandb.log({'test/score': best_score, 'test/safety-layer-usage': sl_usage, 'test/actions_l2_dist': l2_dists})

    # Main loop
    s_t = env.reset().reshape(1, -1)  # pyagents wants first dim to be batch dim
    for epoch in (pbar := tqdm.trange(0, num_epochs, train_steps, desc='TRAINING')):

        # Env interactions
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, done, info = env.step(a_t[0])
            s_tp1 = s_tp1.reshape(1, -1)
            feasible_action = env.rescale(info['action'], to_network_range=True).reshape(1, -1)
            agent.remember(state=s_t,
                           action=feasible_action,
                           reward=np.asarray([r_t]),
                           next_state=s_tp1,
                           done=[done],
                           logprob=lp_t)
            # also store unfeasible action in buffer
            if not info['feasible'] and False:  # TODO check if useful
                # FIXME agents without SL save the same memory twice
                agent.remember(state=s_t,
                               action=a_t,
                               reward=np.asarray([r_t]) * 2.,
                               next_state=s_tp1,
                               done=[done],
                               logprob=lp_t)
            if 'episode' in info:
                episode += 1
                wandb.log({'episode': episode,
                           'train/score': info['episode']['r'],
                           'train/length': info['episode']['l']})
            s_t = env.reset().reshape(1, -1) if done else s_tp1
        # Training
        for _ in range(train_steps):
            loss_dict = agent.train(batch_size)
            pbar.set_postfix(loss_dict)

        # Testing
        if test_env is not None and epoch > k * test_every:
            pbar.set_description('TESTING')
            score, sl_usage, l2_dists = test_agent(agent, test_env, render_plots=False)
            if score > best_score:
                best_score = score
                agent.save(ver=k)
            k += 1
            loss_dict['test/score'] = score
            wandb.log({'test/score': score, 'test/safety-layer-usage': sl_usage, 'test/actions_l2_dist': l2_dists})
            pbar.set_description(f'[EVAL SCORE: {score:4.0f}] TRAINING')

    return agent


########################################################################################################################

def test_agent(agent, test_env, render_plots=True, save_path=None):
    done = False
    s_t = test_env.reset().reshape(1, -1)  # pyagents wants first dim to be batch dim
    score = 0
    all_actions = []
    l2_dists = []

    # Perform an episode
    while not done:
        agent_out = agent.act(s_t)
        a_t, lp_t = agent_out.actions, agent_out.logprobs
        s_tp1, r_t, done, info = test_env.step(a_t[0])
        if 'action' in info:
            a_t = info['action']
        s_tp1 = s_tp1.reshape(1, -1)
        all_actions.append(np.squeeze(a_t))
        l2_dists.append(info['actions_l2_dist'])
        score += r_t
        s_t = test_env.reset().reshape(1, -1) if done else s_tp1

    if render_plots:
        online_heuristic.compute_real_cost(instance_idx=test_env.mr,
                                           predictions_filepath=os.path.join('data', 'Dataset10k.csv'),
                                           shifts_filepath=os.path.join('data', 'optShift.npy'),
                                           prices_filepath=os.path.join('data', 'gmePrices.npy'),
                                           decision_variables=np.array(all_actions),
                                           display=False,
                                           savepath=save_path,
                                           wandb_log=agent.is_logging)
    return score, info['sl_usage'], np.mean(l2_dists)


########################################################################################################################


def min_max_scaler(starting_range: Tuple[Union[float, int]],
                   new_range: Tuple[Union[float, int]],
                   value: float) -> float:
    """
    Scale the input value from a starting range to a new one.
    :param starting_range: tuple of float; the starting range.
    :param new_range: tuple of float; the new range.
    :param value: float; value to be rescaled.
    :return: float; rescaled value.
    """

    assert isinstance(starting_range, tuple) and len(starting_range) == 2, \
        "feature_range must be a tuple as (min, max)"
    assert isinstance(new_range, tuple) and len(new_range) == 2, \
        "feature_range must be a tuple as (min, max)"

    min_start_value = starting_range[0]
    max_start_value = starting_range[1]
    min_new_value = new_range[0]
    max_new_value = new_range[1]

    value_std = (value - min_start_value) / (max_start_value - min_start_value)
    scaled_value = value_std * (max_new_value - min_new_value) + min_new_value

    return scaled_value


########################################################################################################################


def timestamps_headers(num_timeunits: int) -> List[str]:
    """
    Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
    For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
    :param num_timeunits: int; the number of timeunits in a day.
    :return: list of string; list of timeunits.
    """

    start_time = datetime.strptime('00:00', '%H:%M')
    timeunit = 24 * 60 / num_timeunits
    timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
    timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

    return timestamps


########################################################################################################################


def instances_preprocessing(instances: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PV and Load values from string to float.
    :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
    :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
    """

    assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
    assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

    # Instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    # Instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

    return instances
