"""This module contains the objects used to describe games

CardShark uses a state pattern to model games. To implement a new 
game you will need to implement a game class derived from Game which
describes the overall state of the game. 

The actual gameplay is handled by State objects. Each part of 
your game should be modelled using classes that derive from State.
They should implement the State.handle() method to take in an action
and advance the state of your Game. They should then return another
(or the same) state object which tells the engine what the next 
state will be.

The Player class represents a player within your game. You should
implement your own player classes derived from this for your specific
purposes.

A very simple example could be ... TODO: write this
"""

# Python stuff
import itertools
import typing
from abc import ABC, abstractmethod
import numpy as np

# TF agents stuff
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

# other cardshark stuff
from cardshark.named_object import NamedObject

class ActionSpace(NamedObject):
    """Describes the action space of a Game environment
    
    Contains the names of the actions, the limits of the space for each action, and the tensorflow
    action spec needed when constructing TF agents.

    Attributes:
        action_names (list of str): The names of each action
        action_min (list of int): The minimum values each action can take
        action_max (list of int): The maximum values each action can take

        _tf_spec (tf_agents.specs.BoundedArraySpec): The tensorflow action spec used when
        constructing agents and the underlying gym environment
    """

    def __init__(self, spec: typing.Dict[str, typing.Tuple[int]], **kwargs):

        super().__init__(**kwargs)

        self.action_names: typing.List[str] = []
        self.action_min: np.ndarray = np.zeros((len(spec.items())))
        self.action_max: np.ndarray = np.zeros((len(spec.items())))

        for i, (name, limits) in enumerate(spec.items()):
            assert len(limits) == 2, "limits for action " + name + " are the wrong length, should be (min, max)"

            self.action_names.append(name)
            self.action_min[i] = limits[0]
            self.action_max[i] = limits[1]

        print(self.action_min)
        print(self.action_max)

        self._tf_spec = BoundedArraySpec(
            minimum=self.action_min,
            maximum=self.action_max,
            shape=(len(self.action_names), ),
            dtype=np.int32,
        )

    
    def unravel(self):
        """Unravel the action space of the environment

        e.g. if the current action spec is an N by M multidimensional array of actions,
        this fn will transform the space into a one dimensional one of size M x N.
        The structure of the initial higher dimensional space will be stored in the
        _unravelled_action_space attribute. To get from the 1D action space back to the
        original space you can do::

            action = self._unravelled_action_space[1DactionID]

        """

        self.debug("Generating unravelled action space")
        to_product = []
        for min_val, max_val in zip(self._tf_spec.minimum, self._tf_spec.maximum):
            self.debug("    MIN:", min_val, "MAX:", max_val)
            to_product.append(list(range(min_val, max_val)))

        self.debug("  Taking cartesian product of:", to_product)

        self._unravelled_action_space = np.array(
            list(itertools.product(*to_product))
        )

        self.debug("  Unravelled action space:", self._unravelled_action_space)
        self.debug("  Number of possible actions:", len(self._unravelled_action_space))

        self._tf_spec = BoundedArraySpec(
            minimum=0,
            maximum=len(self._unravelled_action_space) - 1,
            shape=(),
            dtype=np.int32,
        )

class ObservationSpace(NamedObject):
    """Describes the observation space of a Game environment

    Contains limits on each observation, names of the observations and the ArraySpec
    needed for constructing tf agents and gym environment

    Attributes:
        observation_names (list of list of str): The names of each dimension of the space
        observation_min (numpy.array): The min values each observation can take
        observation_max (numpy.array): The max values each observation can take
        _tf_spec (tf_agents.specs.BoundedArraySpec): The tensorflow observation spec used when
        constructing agents and the underlying gym environments
    """
    
    def __init__(
            self, 
            min: np.ndarray, 
            max: np.ndarray, 
            action_space: ActionSpace, 
            n_players: int, 
            names: typing.List[typing.List[str]] = None, 
            **kwargs
        ):

        super().__init__(**kwargs)

        assert min.shape == max.shape, "Min and Max arrays must have matching shapes"

        if names is not None:
            assert len(names) == len(min.shape), "Name list shape does not match array shape"

        self.observation_names: typing.List[typing.List[str]] = names
        self.observation_min: np.ndarray = min
        self.observation_max: np.ndarray = max

        self.debug("obs_spec_min: ", min)
        self.debug("obs_spec_max: ", max)

        self._tf_spec = {
            "observations": BoundedArraySpec(
                minimum=min.flatten(),
                maximum=max.flatten(),
                shape=max.flatten().shape,
                name="observation",
                dtype=np.float32,
            ),
            "mask": BoundedArraySpec(
                minimum=0,
                maximum=1,
                shape=(action_space._tf_spec.maximum - action_space._tf_spec.minimum + 1,),
                dtype=np.int32,
                name="mask",
            ),
            "activePlayer": BoundedArraySpec(
                minimum=0,
                maximum=n_players,
                shape=(),
                dtype=np.int32,
                name="activePlayer",
            ),
        }




class Game(py_environment.PyEnvironment, NamedObject, ABC):
    """Base class for user defined Games

    This should describe the state of the game. You'll need to implement 
    the following methods for your game:

        - _reset_game()
        - check_status()
        - _get_mask()
        - _get_observation()
    """

    def __init__(
        self, n_players: int, unravel_action_space: bool, max_steps: int, **kwargs
    ):
        self.n_players = n_players
        self._unravel_action_space = unravel_action_space
        self._max_steps = max_steps

        self._active_player = None
        self._game_state = None
        self._action_space = None
        self._observation_space = None
        self._winner = None
        self._step_count = 0

        self._info = {}
        self._info["winner"] = -999
        self._info["reward"] = 0
        self._info["skippingTurn"] = False

        self.player_list = []

        NamedObject.__init__(self, **kwargs)
        py_environment.PyEnvironment.__init__(self)

    @abstractmethod
    def _reset_game(self):
        """Reset the Game back to "factory settings"
        
        Will get called when starting a new game, and in the
        Game initialiser. You should reset any instance variables that
        you use in describing your game environment here.
        """

    @abstractmethod
    def _check_status(self) -> bool:
        """Check the current status of the Game
        
        This should return True when your game is totally finished
        and False otherwise
        """

    @abstractmethod
    def _get_mask(self, action: np.ndarray, player_id: int) -> np.array:
        """Should return True if the provided action is allowed and False otherwise
        """

    @abstractmethod
    def _get_observation(self, player_id: int) -> np.ndarray:
        """Get the observation representing the current state of the game
        
        This will be passed to the reinforcement learning agents and
        allow them to assess the current situation and decide what 
        action to take. The format of the observation should fit with what 
        was previously defined for this Game.
        """

    def _reset(self):
        """Reset all players, call the user defined _reset_game() and get the reset timestep"""

        ## set individual pieces back to initial state
        for player in self.player_list:
            player.reset()

        self._reset_game()

        return ts.restart(
            observation={
                "observation": self._get_observation(self.get_active_player()),
                "mask": self.get_mask(self.get_active_player()),
                "activePlayer": self.get_active_player(),
            }
        )
    
    def get_mask(self, player_id: int):
        """Get the mask values for all possible actions using the user defined _get_mask()"""

        mask = np.zeros((self._action_space._unravelled_action_space.shape[0]), dtype=np.int32)

        for action_id, action in enumerate(self._action_space._unravelled_action_space):
            if self._get_mask(action, player_id):
                mask[action_id] = 1
            else:
                mask[action_id] = 0

        self.trace("get_mask(): mask:\n", mask)
        return mask

    def set_action_spec(self, spec: typing.Dict[str, typing.Tuple[int]]) -> None:
        """Use this to describe the possible actions for agents playing your game

        Construct an ActionSpace from spec and assign it to the _action_space attribute.

        Args:
          spec (dict of str: tuple of int): Dictionary describing the actions. Keys
          should be the name of the action. Values should be tuple of (min, max) possible
          values for the action
        """
        
        self._action_space = ActionSpace(spec, name = self.name + "_ActionSpec", logger = self.logger)
        
        if self._unravel_action_space:
            self._action_space.unravel()

    def set_observation_spec(self, min: np.ndarray, max: np.ndarray, names: typing.List[typing.List[str]] = None) -> None:
        """Use this to describe the shape of the observations for agents playing your game
        
        Args:
            min (numpy.ndarray): Array describing the minimum values of the observation array
            max (numpy.ndarray): Array describing the maximum values of the observation array
            names (list of list of str): 
        """

        assert self._action_space is not None, "You need to call set_action_spec() before calling set_observation_spec()"
    
        self._observation_space = ObservationSpace(
            min, 
            max, 
            self._action_space, 
            self.n_players, 
            names, 
            name = self.name + "_ObservationSpace", 
            logger = self.logger
        )

    ## for returning general info about the environment, not things necessarily
    ## needed by agents as observations.
    ## TODO: This should be automated. should construct the _info dict from info
    ## set by the user, should add helper fns like set_winner(), skip_turn()...
    ## TODO: Add a give_reward() to Game that keeps track of the reward issued
    ## this turn so it can be added to the _info and user doesn't have to
    ## set this manually
    def get_info(self) -> typing.Dict:
        """Get information about the state of the game

        This info
        """

        return self._info

    def observation_spec(self) -> typing.Dict[str, BoundedArraySpec]:
        """Get the observation specification for the environment associated with this Game

        This is a dictionary with entries:
            - "observations": the ArraySpec describing the observations returned by 
            the environment.
            - "mask":         the ArraySpec describing the mask returned by the environment.
            - "activePlayer": ArraySpec describing the part of the observation that 
            tells which player is active
        """
        return self._observation_space._tf_spec

    def action_spec(self) -> BoundedArraySpec:
        """Get the action specification for the environment associated with this Game object"""
        return self._action_space._tf_spec

    def flatten_action(self, action_ndim):
        """Take an N dimensional action array and find which 1d action index it corresponds to"""

        # could definitely be smarter about this but I'm lazy... :/
        for i in range(self._action_space._unravelled_action_space.shape[0]):
            if np.all(self._action_space._unravelled_action_space[i] == action_ndim):
                return np.array([i])

        raise ValueError(
            "Seems that the specified action ain't in the action space bud"
        )

    def get_active_player(self) -> int:
        """Get the index of the currently active player
        """
        return self._active_player

    def set_active_player(self, player_id: int) -> None:
        """Set the currently active player using their index
        """
        if player_id > len(self.player_list):
            raise ValueError("Index too high! There aren't that many players!")
        if player_id < 0:
            raise ValueError("Negative index! What is a negative player?!?!")

        self._active_player = player_id

    def check_status(self) -> bool:
        """Check the current status of the game. If finished move to reward state
        
        Checks the current state of the game using the user implemented 
        _check_status() method. If it returns True then move to the reward
        state and return true. If game is already in the reward state, stay there
        and return True. Otherwise do nothing and return False.
        """

        if self._game_state == RewardState:
            ## if we've already checked and are finished we can just return here
            return True

        # do the user supplied status check
        # if game is over, reset the active player to 0 and move to reward state
        if self._check_status():
            self._game_state = RewardState
            self._active_player = 0

            return True

        return False
    
    def skipped_turn(self) -> None:
        """Use this to inform the game that the last players turn was skipped

        Not strictly necessary but can be helpful for training agents as it will
        inform them to disregard the skipped turn when training as it won't provide
        them any useful information and might confuse their training algorithm.
        """
        
        self._info["skippingTurn"] = True

    def _step(self, action: np.ndarray) -> None:
        """Step the game forward one iteration"""
        self.info("")
        self.info("##### Stepping :: Step {} #####".format(self._step_count))
        self.debug("gameState:", self._game_state)
        self.debug("specified actions:", action)

        ## set default info values for this step
        self._info["reward"] = 0
        self._info["skippingTurn"] = False

        ## might need to re-ravel the action
        if self._unravel_action_space:
            action = self._action_space._unravelled_action_space[action]
            self.debug("ravelled actions:", action)

        self.debug("Active player", self.player_list[self._active_player])

        ## handle the action and move the game to the new state
        self._game_state = self._game_state.handle(action, self)
        assert issubclass(self._game_state, GameState)

        terminated = self._game_state == TerminatedState
        if not terminated:
            if self.check_status():
                self._game_state = RewardState

        ## if the number of steps has gone above maximum, we'll truncate the game here
        truncated = self._step_count > self._max_steps

        ## set this so that on the outside, we know which player we should give the reward to
        reward = self.player_list[self._active_player].claim_reward()

        self._info["winner"] = self._winner
        self._info["reward"] = reward

        if not terminated:
            if truncated:
                self.info("::::: Game Truncated :::::")
                step = ts.truncation(
                    reward=reward,
                    discount=1.0,
                    observation={
                        "observation": self._get_observation(self._active_player),
                        "mask": self.get_mask(self._active_player),
                        "activePlayer": self._active_player,
                    },
                )

            else:
                step = ts.transition(
                    reward=reward,
                    discount=1.0,
                    observation={
                        "observation": self._get_observation(self._active_player),
                        "mask": self.get_mask(self._active_player),
                        "activePlayer": self._active_player,
                    },
                )

        else:
            self.info("::::: Game Terminated :::::")
            step = ts.termination(
                reward=reward,
                observation={
                    "observation": self._get_observation(self._active_player),
                    "mask": self.get_mask(self._active_player),
                    "activePlayer": self._active_player,
                },
            )

        self._step_count += 1
        return step


class GameState(ABC):
    """Abstract base class representing a game state

    Users should derive states from this object when implementing a game.
    Will need to implement a handle() method which should perform actions 
    specified by a player.
    """

    @staticmethod
    @abstractmethod
    def handle(action: np.ndarray, game: Game):
        """Handle the action
        
        should advance the state of the Game object based 
        on the provided action array.
        """

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of this state, use for logging"""

class RewardState(GameState):
    """Special game state which occurs at the end of play 
    
    Go through all players and hand out final rewards.
    """

    @staticmethod
    def handle(action: np.ndarray, game: Game) -> GameState:
        game.set_active_player((game.get_active_player() + 1) % game.n_players)

        if game.get_active_player() == 0:
            game.info("========= DONE HANDING OUT REWARDS ==========")
            return TerminatedState

        return RewardState

    @staticmethod
    def name() -> str:
        return "Reward"


class TerminatedState(GameState):
    """Special game state used to indicate that the game has been terminated"""

    @staticmethod
    def handle(action: np.ndarray, game: Game) -> GameState:
        raise NotImplementedError(
            "handle() should never actually get called for TerminatedState. "
            "Something has gone wrong!!"
        )

    @staticmethod
    def name() -> str:
        return "Terminated"


class Player(NamedObject, ABC):
    """The base class representing a player in a Game to be implemented by the user

    You should use the give_reward() method within your Game object to reward the player 
    (and thus the Agent that is embodying that Player) which will inform the reinforcement
    learning process when Agents are trying to learn to play your game.
    """

    def __init__(self, **kwargs):
        NamedObject.__init__(self, **kwargs)

        self._reward_accum = 0.0

        self.reset()

    @abstractmethod
    def reset(self):
        """Reset this player back to "factory settings" 

        Should be implemented by the user and should reset all the stuff you've 
        done in the course of running a game. This will typically be called 
        when instantiating and when resetting a Game object. Also called in the 
        __init__() method of Player.
        """

    def give_reward(self, reward: float) -> None:
        """Give reward to this player for a job well done
        """

        self.debug("Giving reward:", reward)
        self._reward_accum += reward
        self.debug("  Reward after:", self._reward_accum)

    def claim_reward(self) -> float:
        """Get the total reward given to this player until now and set it's reward back to 0
        """
        self.debug("Claiming reward:", self._reward_accum)
        ret = self._reward_accum
        self._reward_accum = 0.0
        return ret

    def check_reward(self) -> float:
        """Check the reward that this player has accumulated but without resetting it to 0
        """
        return self._reward_accum

    def get_info_str(self) -> str:
        """Can overwrite this to provide a string that gives information about this Player
        
        This will then be used in some debug printouts and can give some help when trying 
        to debug user code.
        """
        return ""

    def __str__(self) -> str:
        ret_str = "Name: " + self.name + "\n" + self.get_info_str()

        return ret_str
