from Logging import *
import numpy as np
import os
from tensorflow import keras
from tensorflow import function as tfFunction
from tensorflow import Variable
from tf_agents.trajectories import trajectory
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import sequential
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import matplotlib.pyplot as plt
from tf_agents.utils import common

class MultiAgent():
    nAgents = 0

    def __init__(self,
                 ## positional args to pass through to the DqnAgent
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 ## optional args for MultiAgent
                 observation_spec = None,
                 logger: Logger = None,
                 AgentName: str = "",
                 batchSize_train: int = 4,
                 maxBufferLength: int = 1000,
                 applyMask: bool = True,
                 fcLayerParams: tuple = (100,),
                 checkpointPath: str = None,
                 ## optional DqnAgent args
                 **DqnAgent_args
                 ):
        
        if AgentName == "":
            self.Name = "Agent_" + str(MultiAgent.nAgents)
            
        else: 
            self.Name = AgentName

        self.logger = logger

        self._applyMask = applyMask

        self._time_step_spec = time_step_spec
        self.DEBUG("TIME STEP SPEC:", self._time_step_spec)

        if observation_spec == None:
            self._observation_spec = self._time_step_spec.observation
        else:
            self._observation_spec = observation_spec       

        self._action_spec = action_spec
        self.DEBUG("ACTION SPEC SHAPE:", self._action_spec.shape)
        self.DEBUG("ACTION SPEC:", self._action_spec)

        self._build_q_net(fcLayerParams)
        self._step = Variable(0)

        self._agent = categorical_dqn_agent.CategoricalDqnAgent(
            self._time_step_spec,
            self._action_spec,
            categorical_q_network=self.q_net,
            min_q_value=-20,
            max_q_value=20,
            n_step_update=2,
            observation_and_action_constraint_splitter=self.obs_constraint_splitter(),
            gamma=0.99,
            train_step_counter=self._step,
            **DqnAgent_args
        )

        self._agent.initialize()

        self.DEBUG("DQN agent initialised!")
        self.q_net.summary()

        self.randomPolicy = RandomTFPolicy(self._time_step_spec, self._action_spec, observation_and_action_constraint_splitter = self.obs_constraint_splitter())
        
        self.batchSize_train = batchSize_train
        self.currentStep = None
        self.lastStep = None
        self.lastPolicyStep = None

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self._agent.training_data_spec,
            batch_size=1,
            max_length=maxBufferLength
        )

        self.DEBUG("Replay buffer initialised:", self._replay_buffer)
        self.DEBUG("           With data spec:", self._replay_buffer.data_spec)

        self.experienceDataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batchSize_train,
            num_steps=2 + 1).prefetch(3)

        self.experienceIterator = iter(self.experienceDataset)
        
        self.losses = []
        self._gameOutcomes = []
        self._rewards = []

        if checkpointPath == None:
            checkpointPath = os.path.join("Checkpoints", self.Name)

        self._checkpointPath = checkpointPath
        self._trainCheckpointer = common.Checkpointer(
            ckpt_dir = self._checkpointPath,
            max_to_keep = 1,
            agent = self._agent,
            policy = self._agent.policy,
            replay_buffer = self._replay_buffer,
            global_step = self._step
        )

        MultiAgent.nAgents += 1
        
    def save_checkpoint(self):
        ## TODO: make a subclass of the tfagent we want to use which has all internal variables defined as TF variables so the whole thing can be saved to a checkpoint maybe?
        self._trainCheckpointer.save(self._step)

    def load_checkpoint(self):
        self._trainCheckpointer.initialize_or_restore()
        print(self._agent.policy)

    def _registerOutcome(self, outcome: int):
        ## add outcome to internal list of game outcomes (-1 for loss, 0 for inconclusive (game truncated), and +1 for win)
        self._gameOutcomes.append(outcome)

    def registerWin(self):
        self._registerOutcome(+1)

    def registerLoss(self):
        self.registerOutcome(-1)
    
    def registerInconclusive(self):
        self._registerOutcome(0)
    
    ## wrap the logger functions... must be a nicer way of doing this...
    def ERROR(self, *messages): 
        if self.logger != None: self.logger.error("{"+self.Name+"}", *messages)
    def WARN(self, *messages): 
        if self.logger != None: self.logger.warn("{"+self.Name+"}", *messages)
    def INFO(self, *messages): 
        if self.logger != None: self.logger.info("{"+self.Name+"}", *messages)
    def DEBUG(self, *messages): 
        if self.logger != None: self.logger.debug("{"+self.Name+"}", *messages)
    def TRACE(self, *messages): 
        if self.logger != None: self.logger.trace("{"+self.Name+"}", *messages)

    ## Function to extract the observations and mask values to pass to the model
    def obs_constraint_splitter(self):
        if self._applyMask:
            #@tfFunction
            def retFn(observationDict):

                self.TRACE("Observation dict:", observationDict)
                return (observationDict["observations"], observationDict["mask"])

            return retFn
        
        ## can just return none then won't be used when initialising the DQN agent
        else:
            return None

    def _build_q_net(self, fcLayerParams):

        self.q_net = categorical_q_network.CategoricalQNetwork(
            self._observation_spec,
            self._action_spec,
            fc_layer_params=fcLayerParams
        )
        
        return

    def setCurrentStep(self, timeStep: tuple):
        self.lastStep = self.currentStep
        self.currentStep = timeStep

    def getAction(self, timeStep: tuple, collect: bool = False, random: bool = False):
        if random and collect:
            raise ValueError("have specified both collect and random policy, this is invalid")
        if random:
            policy_step = self.randomPolicy.action(self.currentStep)
        elif collect:
            policy_step = self.collect_policy.action(self.currentStep)
        else:
            policy_step = self._agent.policy.action(self.currentStep)

        self.lastPolicyStep = policy_step

        action = policy_step.action

        self.DEBUG("Passing action", action, "to env")
        
        return action

    def addFrame(self):
        if((self.lastStep != None) and (self.currentStep != None) and (self.lastPolicyStep != None)):
            self.DEBUG("Adding Frame")
            self.DEBUG("  last time step:   ", self.lastStep)
            self.DEBUG("  last action:      ", self.lastPolicyStep)
            self.DEBUG("  current time step:", self.currentStep)

            traj = trajectory.from_transition(time_step = self.lastStep, 
                                              action_step = self.lastPolicyStep, 
                                              next_time_step = self.currentStep)

            self.DEBUG("  Trajectory:", traj)
            self._replay_buffer.add_batch(traj)

            self._rewards.append(self.currentStep.reward.numpy()[0])

    def trainAgent(self):
        self.DEBUG("::: Training agent :::", self.Name)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(self.experienceIterator)
        self.DEBUG("  using experience:", experience)
        train_loss = self._agent.train(experience).loss
        self.DEBUG("  Loss:", train_loss)
        self.losses.append(train_loss)

        return train_loss

    def plotTrainingLosses(self):
        plt.plot(list(range(len(self.losses))), self.losses)
        plt.title(self.Name + " Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plotRewards(self):
        plt.plot(list(range(len(self._rewards))), self._rewards)
        plt.title(self.Name + " Rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.show()
        
    def getWinRate(self) -> float:
        if len(self._gameOutcomes) == 0:
            return 0.0
        
        else:
            return np.count(np.array(self._gameOutcomes) == 1) / len(self._gameOutcomes)

            