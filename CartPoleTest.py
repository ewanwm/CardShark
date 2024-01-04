from timeit import default_timer as timer

## TF imports
print('\nImport tensorflow sub-modules')
start = timer()
from tensorflow import keras
from tensorflow import Variable
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
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

def runEpisodeSingleAgent(train: bool, # <- Whether to train the agent during this episode
                          agent, # <- the agent to test
                          env: tf_py_environment.TFPyEnvironment, # <- the environment to run the episode in
                          collect: bool = True, # <- whether to use the collect policy of the agent
                          random: bool = False, # <- whether to use the random policy of the agent
                          toVideo: bool = False, # <- whether to render the episode to a video
                          videoFileName: str = "CartPoleTest.mp4", # <- the name of the file to save the video to
                          videoFPS: int = 24 # <- FPS of saved video
                          ): 
    
    """ fn to run a single episode of the provided environment, either to collect data, train, or evaluate"""
    
    episodePerformance = namedtuple("episodePerformance", "stepCount episodeReward losses nFrames")
    nFrames = 0
    reward = 0.0
    losses = []

    if toVideo:
       video = imageio.get_writer(videoFileName, fps=videoFPS)

    step = env.reset()
    agent.setCurrentStep(step)
    action = agent.getAction(step, collect=collect, random=random)

    if toVideo:
        ## might need to account for the batch dimension
        if env.batched:
            video.append_data(env.pyenv.render()[0])
        else:
            video.append_data(env.pyenv.render())
    
    ## loop until the episode is finished (the env returns a LAST type step)
    while(step.step_type != ts.StepType.LAST):

        step = env.step(action)

        agent.setCurrentStep(step)
        agent.addFrame()

        action = agent.getAction(step, collect=collect, random=random)
        

        #step = agent.step_addFrame(env, collect, random)

        if train: 
            losses.append(agent.trainAgent().numpy())

        if toVideo:
            ## might need to account for the batch dimension
            if env.batched:
                video.append_data(env.pyenv.render()[0])
            else:
                video.append_data(env.pyenv.render())

        reward += step.reward.numpy()[0]
        nFrames += 1

    if toVideo: video.close()

    return episodePerformance(agent.train_step_counter.numpy(), reward, losses, nFrames)

def testMultiAgent():
    
    ## Set up the agent
    agent = MultiAgent.MultiAgent(
            train_env.time_step_spec(), 
            train_env.action_spec(), 
            optimizer = keras.optimizers.Adam(learning_rate=1e-3),
            logger=Logging.Logger(name = "CartPoleTest_Logger", logLevel = Logging.logLevels.kInfo, toFile = True),
            batchSize_train=64,
            epsilon_greedy=0.4,
            applyMask = False,
            td_errors_loss_fn=common.element_wise_squared_loss
        )
    
    ## Make a video of the agents performance pre-training
    perf = runEpisodeSingleAgent(False, agent, eval_env, collect=False, toVideo=True, videoFileName="PreTrainingCartPole.mp4")
    print("Pre Train performance:", perf)

    ## now train the agent
    ## first collect some data without training
    for _ in range(20):
        print("  Collection perf:", runEpisodeSingleAgent(False, agent, train_env, collect = False, random = True))

    print("NFrames collected:", agent._replay_buffer.num_frames())
    ## now actually train
    for _ in range(200):
        print("  Train perf:", runEpisodeSingleAgent(True, agent, train_env))

    agent.plotTrainingLosses()
    agent.plotRewards()

    ## Now make a video of the agents performance post-training
    perf = runEpisodeSingleAgent(False, agent, eval_env, collect=False, toVideo=True, videoFileName="PostTrainingCartPole.mp4")
    print("Post Train performance:", perf)
        
testMultiAgent()
