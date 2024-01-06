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
                 agent_name: str = "",
                 train_batch_size: int = 4,
                 max_buffer_length: int = 1000,
                 apply_mask: bool = True,
                 fc_layer_params: tuple = (100,),
                 checkpoint_path: str = None,
                 ## optional DqnAgent args
                 **DqnAgent_args
                 ):
        
        if agent_name == "":
            self.Name = "Agent_" + str(MultiAgent.nAgents)
            
        else: 
            self.Name = agent_name

        self.logger = logger

        self._apply_mask = apply_mask

        self._time_step_spec = time_step_spec
        self.DEBUG("TIME STEP SPEC:", self._time_step_spec)

        if observation_spec == None:
            self._observation_spec = self._time_step_spec.observation
        else:
            self._observation_spec = observation_spec       

        self._action_spec = action_spec
        self.DEBUG("ACTION SPEC SHAPE:", self._action_spec.shape)
        self.DEBUG("ACTION SPEC:", self._action_spec)

        self._build_q_net(fc_layer_params)
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

        self.random_policy = RandomTFPolicy(self._time_step_spec, self._action_spec, observation_and_action_constraint_splitter = self.obs_constraint_splitter())
        
        self.train_bs = train_batch_size
        self.current_step = None
        self.last_step = None
        self.last_policy_step = None

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self._agent.training_data_spec,
            batch_size=1,
            max_length=max_buffer_length
        )

        self.DEBUG("Replay buffer initialised:", self._replay_buffer)
        self.DEBUG("           With data spec:", self._replay_buffer.data_spec)

        self.experience_dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.train_bs,
            num_steps=2 + 1).prefetch(3)

        self.experience_iterator = iter(self.experience_dataset)
        
        self.losses = []
        self._game_outcomes = []
        self._rewards = []

        if checkpoint_path == None:
            checkpoint_path = os.path.join("Checkpoints", self.Name)

        self._checkpoint_path = checkpoint_path
        self._trainCheckpointer = common.Checkpointer(
            ckpt_dir = self._checkpoint_path,
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

    def _register_outcome(self, outcome: int):
        ## add outcome to internal list of game outcomes (-1 for loss, 0 for inconclusive (game truncated), and +1 for win)
        self._game_outcomes.append(outcome)

    def register_win(self):
        self._register_outcome(+1)

    def register_loss(self):
        self.register_outcome(-1)
    
    def register_inconclusive(self):
        self._register_outcome(0)
    
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
        if self._apply_mask:
            #@tfFunction
            def retFn(observationDict):

                self.TRACE("Observation dict:", observationDict)
                return (observationDict["observations"], observationDict["mask"])

            return retFn
        
        ## can just return none then won't be used when initialising the DQN agent
        else:
            return None

    def _build_q_net(self, fc_layer_params):

        self.q_net = categorical_q_network.CategoricalQNetwork(
            self._observation_spec,
            self._action_spec,
            fc_layer_params=fc_layer_params
        )
        
        return

    def set_current_step(self, timeStep: tuple):
        self.last_step = self.current_step
        self.current_step = timeStep

    def get_action(self, timeStep: tuple, collect: bool = False, random: bool = False):
        if random and collect:
            raise ValueError("have specified both collect and random policy, this is invalid")
        if random:
            policy_step = self.random_policy.action(self.current_step)
        elif collect:
            policy_step = self.collect_policy.action(self.current_step)
        else:
            policy_step = self._agent.policy.action(self.current_step)

        self.last_policy_step = policy_step

        action = policy_step.action

        self.DEBUG("Passing action", action, "to env")
        
        return action

    def add_frame(self):
        if((self.last_step != None) and (self.current_step != None) and (self.last_policy_step != None)):
            self.DEBUG("Adding Frame")
            self.DEBUG("  last time step:   ", self.last_step)
            self.DEBUG("  last action:      ", self.last_policy_step)
            self.DEBUG("  current time step:", self.current_step)

            traj = trajectory.from_transition(time_step = self.last_step, 
                                              action_step = self.last_policy_step, 
                                              next_time_step = self.current_step)

            self.DEBUG("  Trajectory:", traj)
            self._replay_buffer.add_batch(traj)

            self._rewards.append(self.current_step.reward.numpy()[0])

    def train_agent(self):
        self.DEBUG("::: Training agent :::", self.Name)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(self.experience_iterator)
        self.DEBUG("  using experience:", experience)
        train_loss = self._agent.train(experience).loss
        self.DEBUG("  Loss:", train_loss)
        self.losses.append(train_loss)

        return train_loss

    def plot_training_losses(self):
        plt.plot(list(range(len(self.losses))), self.losses)
        plt.title(self.Name + " Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_rewards(self):
        plt.plot(list(range(len(self._rewards))), self._rewards)
        plt.title(self.Name + " Rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.show()
        
    def get_win_rate(self) -> float:
        if len(self._game_outcomes) == 0:
            return 0.0
        
        else:
            return np.count(np.array(self._game_outcomes) == 1) / len(self._game_outcomes)

            