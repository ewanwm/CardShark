from timeit import default_timer as timer

## TF imports
print('\nImport tensorflow')
start = timer()
import tensorflow as tf
print(tf.__version__)
end = timer()
print('Elapsed time: ' + str(end - start))

print('\nImport tensorflow sub-modules')
start = timer()
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
end = timer()
print('Elapsed time: ' + str(end - start))


print('\nimport Coup AI modules')
start = timer()
import MultiAgent
import Logging 
end = timer()
print('Elapsed time: ' + str(end - start))

import PIL
from collections import namedtuple
import base64
import imageio
import IPython
import os
print("Imported other modules")

## Set up the CartPole environment from Gym
env_name = 'CartPole-v0'

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

## Set up the agent
agent = MultiAgent.MultiAgent(
        train_env.time_step_spec(), train_env.action_spec(), 
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        logger=Logging.Logger(name = "CartPoleTest_Logger", logLevel = Logging.logLevels.kInfo, toFile = True),
        batchSize_train=64,
        epsilon_greedy = 0.2,
        applyMask = False
    )

def runEpisodeSingleAgent(train: bool, # <- Whether to train the agent during this episode
                          env: tf_py_environment.TFPyEnvironment, # <- the environment to run the episode in
                          collect: bool = True, # <- whether to use the collect policy of the agent
                          toVideo: bool = False, # <- whether to render the episode to a video
                          videoFileName: str = "CartPoleTest.mp4", # <- the name of the file to save the video to
                          videoFPS: int = 24 # <- FPS of saved video
                          ): 
    
    """ fn to run a single episode of the provided environment, either to collect data, train, or evaluate"""
    
    episodePerformance = namedtuple("episodePerformance", "rewards losses nFrames")
    nFrames = 0
    rewards = []
    losses = []

    if toVideo:
       video = imageio.get_writer(videoFileName, fps=videoFPS)

    ## TODO: add random collection policy to the MultiAgent that can be used to randomly sample environment
    step = env.reset()

    if toVideo: video.append_data(env.render())
    
    ## loop until the episode is finished (the env returns a LAST type step)
    while(step.step_type != ts.StepType.LAST):

        agent.setCurrentStep(step)

        action = agent.getAction(step, collect=collect)
        
        agent.addFrame()

        if train: 
            losses.append(agent.trainAgent())
                
        step = env.step(action)

        if toVideo: video.append_data(env.render())

        rewards.append(step.reward.numpy()[0])
        nFrames += 1

    if toVideo: video.close()

    return episodePerformance(rewards, losses, nFrames)

## Make a video of the agents performance pre-training
perf = runEpisodeSingleAgent(False, eval_env, collect=False, toVideo=False, videoFileName="PreTrainingCartPole.mp4")
print("Pre Train performance:", perf)

## now train the agent
## first collect some data without training
for _ in range(5):
    runEpisodeSingleAgent(False, train_env)
## now actually train
for _ in range(25):
    runEpisodeSingleAgent(True, train_env)

agent.plotTrainingLosses()
agent.plotRewards()

## Now make a video of the agents performance post-training
perf = runEpisodeSingleAgent(False, eval_env, collect=False, toVideo=False, videoFileName="PostTrainingCartPole.mp4")
print("Post Train performance:", perf)
