# Author: Mattia Silvestri

"""
    Main methods to train and test the methods.
"""
from online_heuristic import compute_real_cost
from vpp_envs import SingleStepVPPEnv, MarkovianVPPEnv, MarkovianRlVPPEnv, SingleStepFullRLVPP
import numpy as np
import pandas as pd
import os
import argparse
from typing import Union, List
import wandb
from rl.policy import GaussianPolicy, DirichletPolicy
from rl.agent import OnPolicyAgent
from rl.models import PolicyGradient, A2C
from rl.baselines import SimpleBaseline, Critic

########################################################################################################################


TIMESTEP_IN_A_DAY = 96
METHODS = ['hybrid-single-step', 'hybrid-mdp', 'rl-single-step', 'rl-mdp']


########################################################################################################################


def wandb_wrap(wandb_key, algo, method, instance_idx, log_dir, cfg_dict=None):
    def exec_run(*args, **kwargs):
        if wandb_key:
            run = wandb.init(project="rl-online-offline-opt", entity="mazelions", group=method,
                             dir=log_dir, reinit=True, config=cfg_dict, tags=[f'{instance_idx}'])
            wandb.define_metric('train/loss', step_metric="train_step", summary="min")
            res = algo(*args, **kwargs, wandb_log=True)
            run.finish()
        else:
            res = algo(*args, **kwargs, wandb_log=False)
        return res
    return exec_run


def train_rl_algo(method: str = None,
                  test_split: Union[float, List[int]] = 0.25,
                  num_epochs: int = 1000,
                  noise_std_dev: Union[float, int] = 0.01,
                  batch_size: int = 100,
                  filename: str = None,
                  wandb_log: bool = False):
    """
    Training routing.
    :param method: string; choose among one of the available methods.
    :param test_split: float or list of int; fraction or indexes of the instances to be used for test.
    :param num_epochs: int; number of training epochs.
    :param noise_std_dev: float; standard deviation for the additive gaussian noise.
    :param batch_size: int; batch size.
    :param filename: str,
    :param wandb_log: bool, if True log to WandB
    :return:
    """

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
    if isinstance(test_split, float):
        split_index = int(len(predictions) * (1 - test_split))
        train_predictions = predictions[:split_index]
    elif isinstance(test_split, list):
        split_index = test_split
        train_predictions = predictions.iloc[split_index]
    else:
        raise Exception("test_split must be list of int or float")

    if method == 'hybrid-mdp':
        # Create the environment
        env = MarkovianVPPEnv(predictions=train_predictions,
                              shift=shift,
                              c_grid=c_grid,
                              noise_std_dev=noise_std_dev,
                              savepath=None)
    elif method == 'hybrid-single-step':
        # Create the environment
        env = SingleStepVPPEnv(predictions=train_predictions,
                               shift=shift,
                               c_grid=c_grid,
                               noise_std_dev=noise_std_dev,
                               savepath=None)
    elif method == 'rl-single-step':
        # Create the environment
        env = SingleStepFullRLVPP(predictions=train_predictions,
                                  shift=shift,
                                  c_grid=c_grid,
                                  noise_std_dev=noise_std_dev,
                                  savepath=None)
    elif method == 'rl-mdp':
        # Create the environment
        env = MarkovianRlVPPEnv(predictions=train_predictions,
                                shift=shift,
                                c_grid=c_grid,
                                noise_std_dev=noise_std_dev,
                                savepath=None)

    # Get the actions space
    actions_space = env.action_space.shape
    assert len(actions_space) == 1, "Only single dimension tuples are supported"
    actions_space = actions_space[0]

    # Create and train the RL agent

    baseline = Critic(input_shape=env.observation_space.shape,
                      hidden_units=[32, 32])

    # baseline = SimpleBaseline()
    model = A2C(input_shape=env.observation_space.shape,
                output_dim=actions_space,
                critic=baseline,
                hidden_units=[32, 32])

    '''model = PolicyGradient(input_shape=env.observation_space.shape,
                           output_dim=actions_space,
                           hidden_units=[32, 32])'''

    policy = DirichletPolicy(actions_space)

    agent = OnPolicyAgent(env, policy, model, baseline, standardize_q_vals=True, wandb_log=wandb_log)

    agent.train(num_steps=num_epochs * batch_size,
                render=False,
                gamma=0.99,
                batch_size=batch_size,
                filename=filename)

    # Save the agent's actions
    action_save_name = 'solution' if 'rl' in method else 'cvirt'
    all_actions = agent.test(loadpath=filename, render=False)
    if method == 'rl-single-step':
        storage_in = all_actions[:env.n]
        storage_out = all_actions[env.n:env.n * 2]
        grid_in = all_actions[env.n * 2:env.n * 3]
        diesel_power = all_actions[env.n * 3:]
        all_actions = np.stack([storage_in, storage_out, grid_in, diesel_power], axis=-1)
    solutions_filepath = os.path.join(filename, action_save_name)
    np.save(solutions_filepath, all_actions)

    compute_real_cost(instance_idx=test_split[0],
                      predictions_filepath=predictions_filepath,
                      shifts_filepath=shifts_filepath,
                      prices_filepath=prices_filepath,
                      decision_variables=solutions_filepath + '.npy',
                      display=False,
                      savepath=filename,
                      wandb_log=wandb_log)

########################################################################################################################


if __name__ == '__main__':

    # NOTE: you should set the logging directory and the method
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Logging directory")
    parser.add_argument("--method",
                        type=str,
                        choices=METHODS,
                        help="'hybrid-single-step': this is referred to as 'single-step' in the paper;"
                             + "'hybrid-mdp': this is referred to as 'mdp' in the paper;"
                             + "'rl-single-step': end-to-end RL approach which directly provides the decision "
                             + "variables for all the stages;"
                             + "'rl-mdp': this is referred to as 'rl' in the paper.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument('-wk', '--wandb-key', default='', type=str, help='WandB login key')
    parser.add_argument('--retrain', action='store_true',
                        help='If true, retrain on instances already present in log dir (default: false)')
    args = parser.parse_args()

    LOG_DIR = args.logdir
    METHOD = args.method
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # Randomly choose 100 instances
    np.random.seed(0)
    indexes = np.arange(10000, dtype=np.int32)
    indexes = np.random.choice(indexes, size=100)

    print(indexes)

    # Create the logging directory if does not exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Setup WandB logging
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    # Training routing
    for instance_idx in indexes:
        if str(instance_idx) in os.listdir(LOG_DIR) and not args.retrain:
            print(f'Skipping instance {instance_idx}')
        else:
            print(f'Instance index: {instance_idx}')
            run = wandb_wrap(args.wandb_key, train_rl_algo, METHOD,
                             instance_idx=instance_idx, log_dir=LOG_DIR,
                             cfg_dict={'epochs': EPOCHS, 'batch size': BATCH_SIZE})
            run(method=METHOD,
                test_split=[instance_idx],
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                noise_std_dev=0.01,
                filename=os.path.join(LOG_DIR, str(instance_idx)))
            print('-' * 50 + '\n')
