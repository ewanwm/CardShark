# Python stuff
import cProfile
import time
import random

# tensorflow stuff
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents import environments

# cardshark stuff
from cardshark import logging as log
from cardshark.agent import MultiAgent

from examples.coup import coup_engine, coup_agent

nPlayers = 4

## Build the environments for training and evaluation
train_logger = log.Logger(
    name="Train_env_Logger", log_level=log.LogLevel.INFO, to_file=False
)
train_py_env = coup_engine.CoupGame(
    nPlayers=nPlayers, unravelActionSpace=True, logger=train_logger, maxSteps=250
)
batched_train_py_env = environments.BatchedPyEnvironment([train_py_env])
train_env = tf_py_environment.TFPyEnvironment(batched_train_py_env, check_dims=True)

eval_logger = log.Logger(
    name="Eval_env_Logger", log_level=log.LogLevel.INFO, to_file=False
)
eval_py_env = coup_engine.CoupGame(
    nPlayers=nPlayers, unravelActionSpace=True, logger=eval_logger, maxSteps=250
)
batched_eval_py_env = environments.BatchedPyEnvironment([eval_py_env])
eval_env = tf_py_environment.TFPyEnvironment(batched_eval_py_env, check_dims=True)


def runMatchSingleHuman(
    human_agent, robot_agents, collect=False, random=False, train=True
):
    """run a match and train agent"""
    ## TODO: add random collection policy to the MultiAgent that can be used to randomly sample environment
    step = train_env.reset()

    while step.step_type != ts.StepType.LAST:
        ## First player is played by the human
        if step.observation["activePlayer"] == 0:
            action = human_agent.get_action()
            step = train_env.step(action)

        ## for all others we use a random policy
        else:
            ## First player is played by the actual agent
            active_agent = step.observation["activePlayer"][0]

            agent = robot_agents[active_agent]

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

        if train_py_env.player_list[0].isAlive:
            time.sleep(3)
        else:
            time.sleep(0.25)


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


human_agent = coup_agent.CoupHumanAgent(
    name=coup_agent.MAGENTA + "Kyle" + coup_agent.RESET
)
human_agent.register_game(train_py_env, 0)
train_py_env.player_list[0].name = human_agent.name

## generate some random colours for the bots
bot_colours = []
for i in range(nPlayers - 1):
    bot_colours.append("\u001b[38;5;{}m".format(random.randint(21, 230)))
    train_py_env.player_list[1 + i].name = (
        bot_colours[i] + "bot_" + str(i) + coup_agent.RESET
    )

## Load up some (possibly pre-trained) bots to play against the human
robot_agents = []
for i in range(nPlayers):
    robot_agents.append(
        MultiAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            train_batch_size=64,
            observation_spec=train_env.time_step_spec().observation["observations"],
            checkpoint=False,
            DqnAgent_args={
                "optimizer": tf.keras.optimizers.Adam(learning_rate=0.5e-2),
                "epsilon_greedy": 0.4,
            },
            fc_layer_params=(100, 100, 100),
            logger=train_py_env.logger,
            name="bot_" + str(i),
        )
    )


def trainMultiAgent(agents, nMatches, plotInterval):
    step = train_env.reset()

    for collectMatchIndex in range(5):
        log.info("######## Collect Match {} ##########".format(collectMatchIndex))
        runMatchMultiAgent(agents, train=False, random=True)

    for matchIndex in range(1, nMatches):
        log.info("######## Train Match {} ##########".format(matchIndex))
        runMatchMultiAgent(agents, train=True, collect=True)

        if (matchIndex % plotInterval) == 0:
            for agent in agents:
                agent.plot_training_losses()
                agent.plot_rewards()


# train the bots a little bit first
train_loggerlogLevek = log.LogLevel.SILENT  ## sshhhh
trainMultiAgent(robot_agents, 5, 4)

train_logger.log_level = log.LogLevel.INFO
while True:
    runMatchSingleHuman(human_agent, robot_agents)

    play_again = input("Play again? [Y/N]")

    if play_again in ["Y", "y", "Yes", "yes"]:
        continue
    elif play_again in ["N", "n", "No", "no"]:
        break
    else:
        print("Invalid option")
        play_again = input("Play again? [Y/N]")

for agent in robot_agents:
    agent.plot_training_losses()
