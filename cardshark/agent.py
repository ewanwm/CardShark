"""Agents are "things that play games".

These can either be human players, represented by the HumanAgent class,
or machine learning based agents represented by the MultiAgent class.
Within the CardShark engine, agents are in charge of keeping track of
their previous performance and, in the case of ML based agents, their 
own training data. They are also in charge of performing their own training
using that data.

You can also use the AgentBase class to implement your own agent classes 
if needed.
"""

# python stuff
from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt

# TF stuff
from tensorflow import function as tfFunction
from tensorflow import Variable
from tensorflow import convert_to_tensor

# TF agents stuff
from tf_agents.trajectories import trajectory, PolicyStep
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

# other cardshark stuff
from cardshark.named_object import NamedObject
from cardshark.engine import Game


class AgentBase(NamedObject, ABC):
    """Base class for "agents" i.e. a thing that takes observarions and returns an action.

    This could be either a human player or an ML based agent
    """

    def __init__(self, **kwargs):
        NamedObject.__init__(self, **kwargs)

        self._game_outcomes = []

    def _register_outcome(self, outcome: int):
        """Add outcome to internal running list of game outcomes
         
        (-1 for loss, 0 for inconclusive (game truncated), and +1 for win)
        """
        self._game_outcomes.append(outcome)

    def register_win(self):
        """Register that the agent won a game
        """
        self._register_outcome(+1)

    def register_loss(self):
        """Register that the agent lost a game
        """
        self._register_outcome(-1)

    def register_inconclusive(self):
        """Register that the agent took part in a game that ended inconclusively
        """
        self._register_outcome(0)

    def get_win_rate(self) -> float:
        """Get the fraction of games this agent has won over the total number played
        """

        # shortcut for agent that hasn't played any games yet
        if len(self._game_outcomes) == 0:
            return 0.0

        wins = np.count(np.array(self._game_outcomes) == 1)
        total = len(self._game_outcomes)

        return wins / total

    @abstractmethod
    def get_action(self):
        """Should take in an obervation and return an action"""


class HumanAgent(AgentBase, ABC):
    """Base class to be used to represent a human player

    User will need to implement the functionality to give game specific info to the player
    and get a game specific action from them
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._game = None
        self._player_id = None

    def get_action(self):
        """Get the action being performed by a player.

        Take the user defined numpy array action from the _get_action() fn
        and convert it to a format that the Game gym environment will like
        """
        action_ndim = self._get_action()

        action_1dim = self._game.flatten_action(action_ndim)

        action_tuple = PolicyStep()
        action_tuple = action_tuple._replace(action=convert_to_tensor(action_1dim))

        return action_tuple

    @abstractmethod
    def _get_action(self) -> np.ndarray:
        """Return the action being performed by the human player

        This should be in the form of a numpy array which matches the structure of
        the action array used in the Game class that is being played
        """

    def register_game(self, game: Game, player_id: int):
        """Set the Game being played by this player and index of this player within that game.

        TODO: Make this more automated... it's a bit sloppy
        """

        self._game = game
        self._player_id = player_id


class MultiAgent(AgentBase):
    """Machine learning agent that is to be used in multi-agent environments
    
    Keeps track of it's own training data and outcomes of played games.
    """

    def __init__(
        self,
        ## positional args to pass through to the DqnAgent
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        ## optional args for MultiAgent
        observation_spec=None,
        train_batch_size: int = 4,
        max_buffer_length: int = 1000,
        apply_mask: bool = True,
        fc_layer_params: tuple = (100,),
        checkpoint=False,
        checkpoint_path: str = None,
        ## optional DqnAgent args
        DqnAgent_args=None,
        **kwargs,
    ):
        AgentBase.__init__(self, **kwargs)

        ## Set specified properties
        self._apply_mask = apply_mask
        self.train_bs = train_batch_size

        self._time_step_spec = time_step_spec
        self.debug("TIME STEP SPEC:", self._time_step_spec)

        self._action_spec = action_spec
        self.debug("ACTION SPEC:", self._action_spec)

        ## Set optional values
        if observation_spec is None:
            self._observation_spec = self._time_step_spec.observation
        else:
            self._observation_spec = observation_spec

        if checkpoint_path is None:
            checkpoint_path = os.path.join("Checkpoints", self.name)

        self._checkpoint_path = checkpoint_path

        ## Set inital values
        self.losses = []
        self._rewards = []
        self._step = Variable(0)

        self.random_policy = RandomTFPolicy(
            self._time_step_spec,
            self._action_spec,
            observation_and_action_constraint_splitter=self._obs_constraint_splitter(),
        )
        self.current_step = None
        self.last_step = None
        self.last_policy_step = None

        ## Build the agent
        self._build_q_net(fc_layer_params)

        self._agent = categorical_dqn_agent.CategoricalDqnAgent(
            self._time_step_spec,
            self._action_spec,
            categorical_q_network=self.q_net,
            min_q_value=-20,
            max_q_value=20,
            n_step_update=2,
            observation_and_action_constraint_splitter=self._obs_constraint_splitter(),
            gamma=0.99,
            train_step_counter=self._step,
            **DqnAgent_args,
        )

        self._agent.initialize()

        self.debug("DQN agent initialised!")
        self.q_net.summary()

        ## Build the replay buffer
        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self._agent.training_data_spec,
            batch_size=1,
            max_length=max_buffer_length,
        )

        self.debug("Replay buffer initialised:", self._replay_buffer)
        self.debug("           With data spec:", self._replay_buffer.data_spec)

        self.experience_dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.train_bs, num_steps=2 + 1
        ).prefetch(3)

        self.experience_iterator = iter(self.experience_dataset)

        ## make the checkpointer
        if checkpoint:
            self._train_checkpointer = common.Checkpointer(
                ckpt_dir=self._checkpoint_path,
                max_to_keep=1,
                agent=self._agent,
                policy=self._agent.policy,
                replay_buffer=self._replay_buffer,
                global_step=self._step,
            )

        ## Wrap these in tf functions to speed things up a bit
        self._agent.train = common.function(self._agent.train)

    @tfFunction
    def _random_action(self, time_step):
        """Get a random action
        """
        return self.random_policy.action(time_step)

    @tfFunction
    def _collect_action(self, time_step):
        """Get an action decided by the agents collect policy
        """
        return self._agent.collect_policy.action(time_step)

    @tfFunction
    def _inference_action(self, time_step):
        """Get an action decided using the agents trained policy
        """
        return self._agent.policy(time_step)

    def save_checkpoint(self):
        """Save a checkpoint of the agents internal training state

        This includes the model parameters, training state etc.
        
        TODO: make a subclass of the tfagent we want to use which has all internal 
        variables defined as TF variables so the whole thing can be saved to a 
        checkpoint maybe?
        """
        self._train_checkpointer.save(self._step)

    def load_checkpoint(self):
        """Load agents internal state from checkpoint
        """
        self._train_checkpointer.initialize_or_restore()

    ## Function to extract the observations and mask values to pass to the model
    def _obs_constraint_splitter(self):
        """Utility fn to construct the "observation constraint splitter to pass to the q agent

        This is used to split the observation before passing to the network.
        This is needed if the environment provides a mask.
        """

        if self._apply_mask:
            # @tfFunction
            def ret_fn(obs_dict):
                self.trace("Observation dict:", obs_dict)
                return (obs_dict["observations"], obs_dict["mask"])

            return ret_fn

        ## if not _apply_mask can just return none then won't be used when initialising DQN agent
        return None

    def _build_q_net(self, fc_layer_params):
        self.q_net = categorical_q_network.CategoricalQNetwork(
            self._observation_spec, self._action_spec, fc_layer_params=fc_layer_params
        )

    def _set_current_step(self, step: tuple):
        self.last_step = self.current_step
        self.current_step = step

    def get_action(self, collect: bool = False, random: bool = False):
        if random and collect:
            raise ValueError(
                "have specified both collect and random policy, this is invalid"
            )
        if random:
            policy_step = self._random_action(self.current_step)
        elif collect:
            policy_step = self._collect_action(self.current_step)
        else:
            policy_step = self._inference_action(self.current_step)

        self.last_policy_step = policy_step

        action = policy_step.action

        self.debug("Passing action", action, "to env")

        return action

    def add_frame(self):
        """Add the currently stored trajectory to the agents training experience buffer

        The trajectory describes the transition from the last registered environment step
        to the current one and the action the agent took to get here. 
        """
        if (
            (self.last_step is not None)
            and (self.current_step is not None)
            and (self.last_policy_step is not None)
        ):
            self.debug("Adding Frame")
            self.debug("  last time step:   ", self.last_step)
            self.debug("  last action:      ", self.last_policy_step)
            self.debug("  current time step:", self.current_step)

            traj = trajectory.from_transition(
                time_step=self.last_step,
                action_step=self.last_policy_step,
                next_time_step=self.current_step,
            )

            self.debug("  Trajectory:", traj)
            self._replay_buffer.add_batch(traj)

            self._rewards.append(self.current_step.reward.numpy()[0])

    @tfFunction
    def _train_agent(self):
        self.debug("::: Training agent :::", self.name)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, _ = next(self.experience_iterator)
        self.debug("  using experience:", experience)
        train_loss = self._agent.train(experience).loss
        self.debug("  Loss:", train_loss)

        return train_loss

    def train_agent(self):
        """Run an iteration of the training algorithm using experience gained by the agent so far
        """
        loss = self._train_agent()
        self.losses.append(loss)

        return loss

    def plot_training_losses(self):
        """Make a plot of the training losses achieved by this agent so far
        """
        plt.plot(list(range(len(self.losses))), self.losses)
        plt.title(self.name + " Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_rewards(self):
        """Make a plot of the rewards gained by this agent so far
        """
        plt.plot(list(range(len(self._rewards))), self._rewards)
        plt.title(self.name + " Rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.show()
