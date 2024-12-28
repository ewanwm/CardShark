from timeit import default_timer as timer

## TF imports
print("\nImport tensorflow sub-modules")
start = timer()
from tensorflow import keras
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

end = timer()
print("Elapsed time: " + str(end - start))


print("\nimport Coup AI modules")
start = timer()
from cardshark.agent import MultiAgent
from cardshark import logging

end = timer()
print("Elapsed time: " + str(end - start))

from collections import namedtuple
import imageio

print("Imported other modules")

## Set up the CartPole environment from Gym
env_name = "CartPole-v0"

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


def runEpisodeSingleAgent(
    train: bool,  # <- Whether to train the agent during this episode
    agent,  # <- the agent to test
    env: tf_py_environment.TFPyEnvironment,  # <- the environment to run the episode in
    collect: bool = True,  # <- whether to use the collect policy of the agent
    random: bool = False,  # <- whether to use the random policy of the agent
    toVideo: bool = False,  # <- whether to render the episode to a video
    videoFileName: str = "CartPoleTest.mp4",  # <- the name of the file to save the video to
    videoFPS: int = 24,  # <- FPS of saved video
):
    """fn to run a single episode of the provided environment, either to collect data, train, or evaluate"""

    episodePerformance = namedtuple(
        "episodePerformance", "stepCount episodeReward min_loss max_loss nFrames"
    )
    nFrames = 0
    reward = 0.0
    losses = []

    if toVideo:
        video = imageio.get_writer(videoFileName, fps=videoFPS)

    step = env.reset()
    agent._set_current_step(step)
    action = agent.get_action(collect=collect, random=random)

    if toVideo:
        ## might need to account for the batch dimension
        if env.batched:
            video.append_data(env.pyenv.render()[0])
        else:
            video.append_data(env.pyenv.render())

    ## loop until the episode is finished (the env returns a LAST type step)
    while step.step_type != ts.StepType.LAST:
        # step = agent.step_environment(env, save_frame = True, train = train, collect = collect, random = random)

        step = env.step(action)

        agent._set_current_step(step)
        agent.add_frame()

        action = agent.get_action(collect=collect, random=random)

        if train:
            losses.append(agent.train_agent().numpy())

        if toVideo:
            ## might need to account for the batch dimension
            if env.batched:
                video.append_data(env.pyenv.render()[0])
            else:
                video.append_data(env.pyenv.render())

        reward += step.reward.numpy()[0]
        nFrames += 1

    if toVideo:
        video.close()

    if len(losses) == 0:
        min_loss = None
        max_loss = None
    else:
        min_loss = min(losses)
        max_loss = max(losses)

    return episodePerformance(agent._step.numpy(), reward, min_loss, max_loss, nFrames)


def testMultiAgent(init_collect_episodes=20, train_episodes=200, log_interval=10):
    print("TS Spec:", train_env.time_step_spec())
    ## Set up the agent
    agent = MultiAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        train_batch_size=64,
        observation_spec=train_env.time_step_spec().observation,
        apply_mask=False,
        DqnAgent_args={
            "optimizer": keras.optimizers.Adam(learning_rate=1e-3),
            "td_errors_loss_fn": common.element_wise_squared_loss,
        },
        # logger=Logging.Logger(name = "CartPoleTest_Logger", log_level = Logging.LogLevel.DEBUG, to_file = True),
    )

    ## Make a video of the agents performance pre-training
    perf = runEpisodeSingleAgent(
        False,
        agent,
        eval_env,
        collect=False,
        toVideo=True,
        videoFileName="PreTrainingCartPole.mp4",
    )
    print("Pre Train performance:", perf)

    ## now train the agent
    ## first collect some data without training
    print(":::: Collecting some random initial training data ::::")
    for ep in range(1, init_collect_episodes + 1):
        perf = runEpisodeSingleAgent(
            False, agent, train_env, collect=False, random=True
        )
        if ep % log_interval == 0:
            print("  Collection episode {} performance:".format(ep), perf)

    print()

    print(":::: Training the agent ::::")
    ## now actually train
    for ep in range(1, train_episodes + 1):
        perf = runEpisodeSingleAgent(True, agent, train_env, collect=True)
        if ep % log_interval == 0:
            print("  Train episode {} performance:".format(ep), perf)

    print()

    agent.plot_training_losses()
    agent.plot_rewards()

    ## Now make a video of the agents performance post-training
    perf = runEpisodeSingleAgent(
        False,
        agent,
        eval_env,
        collect=False,
        toVideo=True,
        videoFileName="PostTrainingCartPole.mp4",
    )
    print("Post Train performance:", perf)


testMultiAgent()
