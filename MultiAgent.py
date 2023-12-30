from Logging import *
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks import sequential
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import matplotlib.pyplot as plt



class MultiAgent(DqnAgent):
    nAgents = 0

    def __init__(self,
                 ## positional args to pass through to the DqnAgent
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 optimizer: types.Optimizer,
                 logger: Logger = None,
                 AgentName: str = "",
                 ## optional args for MultiAgent
                 batchSize_train: int = 4,
                 maxBufferLength: int = 1000,
                 applyMask: bool = True,
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
       

        self._action_spec = action_spec
        self.DEBUG("ACTION SPEC SHAPE:", self._action_spec.shape)
        self.DEBUG("ACTION SPEC:", self._action_spec)

        self._build_q_net()

        self.losses = []

        DqnAgent.__init__(
            self,
            time_step_spec = self._time_step_spec, 
            action_spec = self._action_spec, 
            q_network = self.q_net, 
            observation_and_action_constraint_splitter = self.obs_constraint_splitter(),
            optimizer = optimizer, 
        )

        self.DEBUG("DQN agent initialised!")
        self.q_net.summary()
        
        self.batchSize_train = batchSize_train
        self.currentStep = None
        self.lastStep = None
        self.lastPolicyStep = None
        self.trajBuffer = []

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self.training_data_spec,
            batch_size=1,
            max_length=maxBufferLength
        )

        self.DEBUG("Replay buffer initialised:", self._replay_buffer)
        self.DEBUG("           With data spec:", self._replay_buffer.data_spec)

        self.experienceDataset = self._replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=self.batchSize_train,
        num_steps=2).prefetch(3)

        self.experienceIterator = iter(self.experienceDataset)

        self._gameOutcomes = []
        self._rewards = []

        MultiAgent.nAgents += 1
        
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
            def retFn(observationDict):

                self.TRACE("Observation dict:", observationDict)
                return (observationDict["observations"], observationDict["mask"]) ##flattenedActionMask)

            return retFn
        
        ## can just return none then won't be used when initialising the DQN agent
        else:
            return None
    
    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def _build_q_net(self):

        fc_layer_params = (100,100)

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]
        
        q_values_layer = tf.keras.layers.Dense(
            self._action_spec.maximum +1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        
        self.q_net = sequential.Sequential(dense_layers + [q_values_layer])
        
        return

    def setCurrentStep(self, timeStep: tuple):
        self.lastStep = self.currentStep
        self.currentStep = timeStep

    def getAction(self, timeStep: tuple, collect: bool = False):
        if collect:
            policy_step = self.collect_policy.action(self.currentStep)
        else:
            policy_step = self.policy.action(self.currentStep)

        self.lastPolicyStep = policy_step

        action = policy_step.action
        
        self.DEBUG("Passing action", action, "to env")
        
        return action
    
    def _addBatch(self):
        self._replay_buffer.add_batch(self.trajBuffer)

    def addFrame(self):
        if((self.lastStep != None) and (self.currentStep != None) and (self.lastPolicyStep != None)):
            self.TRACE("Adding Batch")
            self.TRACE("  last time step:   ", self.lastStep)
            self.TRACE("  last action:      ", self.lastPolicyStep)
            self.TRACE("  current time step:", self.currentStep)

            traj = trajectory.from_transition(time_step = self.lastStep, 
                                              action_step = self.lastPolicyStep, 
                                              next_time_step = self.currentStep)

            self._replay_buffer.add_batch(traj)

            self._rewards.append(self.currentStep.reward.numpy()[0])

    def trainAgent(self):
        self.INFO("Training agent", self.name)

         # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(self.experienceIterator)
        train_loss = self.train(experience).loss
        self.losses.append(train_loss)

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

            