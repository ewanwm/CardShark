# Python stuff
import numpy as np
import cProfile

# tensorflow stuff
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils import common
from tf_agents import environments

# Coup AI stuff
import cardshark
from cardshark import logging as log
from cardshark.agent import MultiAgent

from examples.coup import coup_engine, coup_player

log.INFO("TF Version: ", tf.__version__)

log.INFO("All modules loaded")

log.INFO("Physical Devices:\n", tf.config.list_physical_devices(), "\n\n")

nPlayers = 4

## Build the environments for training and evaluation
train_logger = None  # log.Logger(name = "Train_env_Logger", logLevel = log.logLevels.kInfo, toFile = True)
train_py_env = coup_engine.CoupGame(
    nPlayers=nPlayers, unravelActionSpace=True, logger=train_logger, maxSteps=250
)
batched_train_py_env = environments.BatchedPyEnvironment([train_py_env])
train_env = tf_py_environment.TFPyEnvironment(batched_train_py_env, check_dims=True)

eval_logger = None  # log.Logger(name = "Eval_env_Logger", logLevel = log.logLevels.kInfo, toFile = True)
eval_py_env = coup_engine.CoupGame(
    nPlayers=nPlayers, unravelActionSpace=True, logger=eval_logger, maxSteps=250
)
batched_eval_py_env = environments.BatchedPyEnvironment([eval_py_env])
eval_env = tf_py_environment.TFPyEnvironment(batched_eval_py_env, check_dims=True)


def runMatchSingleAgent(
    agent, train: bool, collect: bool = False, random: bool = False
):
    """run a match and train agent"""
    ## TODO: add random collection policy to the MultiAgent that can be used to randomly sample environment
    step = train_env.reset()

    random_policy = RandomTFPolicy(
        eval_env.time_step_spec(),
        eval_env.action_spec(),
        observation_and_action_constraint_splitter=agent._obs_constraint_splitter(),
    )

    ## wrap up the action to speed things up
    random_action = common.function(random_policy.action)

    while step.step_type != ts.StepType.LAST:
        ## First player is played by the actual agent
        if step.observation["activePlayer"] == 0:
            ## check whether or not we should store this frame
            save_frame = (not train_env.get_info()["skippingTurn"]) or (
                train_env.get_info()["reward"] != 0.0
            )

            agent._set_current_step(step)

            if save_frame:
                agent.add_frame()

            action = agent.get_action(collect=collect, random=random)
            step = train_env.step(action)

            if train:
                agent.train_agent()

        ## for all others we use a random policy
        else:
            action = random_action(step)
            step = train_env.step(action)


def trainSingleAgent(agent, nMatches, plotInterval):
    step = train_env.reset()

    for collectMatchIndex in range(5):
        log.INFO("######## Collect Match {} ##########".format(collectMatchIndex))
        runMatchSingleAgent(agent, train=False, random=True)

    for matchIndex in range(1, nMatches):
        log.INFO("######## Train Match {} ##########".format(matchIndex))
        runMatchSingleAgent(agent, train=True, collect=True)

        if (matchIndex % plotInterval) == 0:
            agent.plot_training_losses()
            agent.plot_rewards()


## Train a single agent among random agents
single_agent = MultiAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    train_batch_size=64,
    observation_spec=train_env.time_step_spec().observation["observations"],
    checkpoint=True,
    DqnAgent_args={
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.5e-2),
        "epsilon_greedy": 0.4,
    },
    fc_layer_params=(100, 100, 100),
    logger=train_py_env.logger,
)

# single_agent.save_checkpoint()

# cProfile.run("trainSingleAgent(single_agent, 50, 50)", filename = "train-profile", sort= 2)
# single_agent.save_checkpoint()
# single_agent.plot_training_losses()
# single_agent.plot_rewards()


########################################
## Now train multiple agents at once! ##
########################################


def runMatchMultiAgent(
    agents, train: bool, collect: bool = False, random: bool = False
):
    """run a match and train agent"""
    ## TODO: add random collection policy to the MultiAgent that can be used to randomly sample environment

    assert len(agents) == nPlayers

    step = train_env.reset()

    while step.step_type != ts.StepType.LAST:
        ## First player is played by the actual agent
        active_agent = step.observation["activePlayer"][0]

        agent = agents[active_agent]

        ## check whether or not we should store this frame
        save_frame = (not train_env.get_info()["skippingTurn"]) or (
            train_env.get_info()["reward"] != 0.0
        )

        agent._set_current_step(step)

        if save_frame:
            agent.add_frame()

        action = agent.get_action(collect=collect, random=random)
        step = train_env.step(action)

        if train:
            agent.train_agent()


def trainMultiAgent(agents, nMatches, plotInterval):
    step = train_env.reset()

    for collectMatchIndex in range(5):
        log.INFO("######## Collect Match {} ##########".format(collectMatchIndex))
        runMatchMultiAgent(agents, train=False, random=True)

    for matchIndex in range(1, nMatches):
        log.INFO("######## Train Match {} ##########".format(matchIndex))
        runMatchMultiAgent(agents, train=True, collect=True)

        if (matchIndex % plotInterval) == 0:
            agent.plot_training_losses()
            agent.plot_rewards()


## Train a single agent among random agents
agent_list = []
for i in range(nPlayers):
    agent_list.append(
        MultiAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            train_batch_size=64,
            observation_spec=train_env.time_step_spec().observation["observations"],
            checkpoint=True,
            DqnAgent_args={
                "optimizer": tf.keras.optimizers.Adam(learning_rate=0.5e-2),
                "epsilon_greedy": 0.4,
            },
            fc_layer_params=(100, 100, 100),
            logger=train_py_env.logger,
            name="bot_" + str(i),
        )
    )

trainMultiAgent(agent_list, 500, 500)
for agent in agent_list:
    agent.save_checkpoint()
    agent.plot_training_losses()
    agent.plot_rewards()
