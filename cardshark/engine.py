# other cardshark stuff
from cardshark.logging import *
from cardshark.named_object import NamedObject

# TF agents stuff
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

# Python stuff
import itertools
import typing
import numpy as np
from abc import ABC, abstractmethod


class Game(py_environment.PyEnvironment, NamedObject, ABC):
    def __init__(
        self, nPlayers: int, unravelActionSpace: bool, maxSteps: int, **kwargs
    ):
        self.nPlayers = nPlayers

        NamedObject.__init__(self, **kwargs)

    ## for returning general info about the environment, not things necessarily needed by agents as observations
    def get_info(self) -> typing.Dict:
        """Get information about the state of the game

        This info
        """

        return self._info

    def observation_spec(self) -> typing.Dict[str, BoundedArraySpec]:
        """Get the observation specification for the environment associated with this Game object

        This is a dictionary with entries:
            - "observations": the ArraySpec describing the observations returned by the environment.
            - "mask":         the ArraySpec describing the mask returned by the environment.
            - "activePlayer": ArraySpec describing the part of the observation that tells which player is active
        """
        return self._observation_spec

    def action_spec(self) -> BoundedArraySpec:
        """Get the action specification for the environment associated with this Game object"""
        return self._action_spec

    def flatten_action(self, action_ndim):
        """Take an N dimensional action array and find which 1d action index it corresponds to"""

        # could definitely be smarter about this but I'm lazy... :/
        for i in range(self._unravelled_action_space.shape[0]):
            if np.all(self._unravelled_action_space[i] == action_ndim):
                return np.array([i])

        raise ValueError(
            "Seems that the specified action ain't in the action space bud"
        )

    def _unravel_action_space(self):
        """Unravel the action space of the environment

        e.g. if the current action spec is an N by M multidimensional array of actions,
        this fn will transform the space into a one dimensional one of size M x N.
        The structure of the initial higher dimensional space will be stored in the
        _unravelled_action_space attribute. To get from the 1D action space back to the
        original space you can do::

            action = self._unravelled_action_space[1DactionID]

        """

        self.DEBUG("Generating unravelled action space")
        toProduct = []
        for min, max in zip(self._action_spec.minimum, self._action_spec.maximum):
            self.DEBUG("    MIN:", min, "MAX:", max)
            toProduct.append([i for i in range(min, max)])

        self.DEBUG("  Taking cartesian product of:", toProduct)

        self._unravelled_action_space = np.array(
            [i for i in itertools.product(*toProduct)]
        )

        self.DEBUG("  Unravelled action space:", self._unravelled_action_space)
        self.DEBUG("  Number of possible actions:", len(self._unravelled_action_space))

        self._action_spec = BoundedArraySpec(
            minimum=0,
            maximum=len(self._unravelled_action_space) - 1,
            shape=(),
            dtype=np.int32,
        )

        return

    @abstractmethod
    def _reset():
        pass

    def getActivePlayer(self) -> int:
        """Get the ID of the currently active player"""
        return int(self.activePlayer)

    @abstractmethod
    def _checkStatus(self) -> bool:
        pass

    def checkStatus(self) -> bool:
        if self.gameState == RewardState:
            ## if we've already checked and are finished we can just return here
            return True

        # do the user supplied status check
        # if game is over, reset the active player to 0 and move to reward state
        if self._checkStatus():
            self.gameState = RewardState
            self.activePlayer = 0

            return True

        return False

    @abstractmethod
    def get_mask(self, playerIdx: int) -> np.array:
        pass

    @abstractmethod
    def getObservation(self, playerIdx: int) -> np.ndarray:
        pass

    def _step(self, action: np.ndarray) -> None:
        """Step the game forward one iteration"""
        self.INFO("")
        self.INFO("##### Stepping :: Step {} #####".format(self._stepCount))
        self.DEBUG("gameState:", self.gameState)
        self.DEBUG("specified actions:", action)

        ## set default info values for this step
        self._info["reward"] = 0
        self._info["skippingTurn"] = False

        ## might need to re-ravel the action
        if self._unravelActionSpace:
            action = self._unravelled_action_space[action]
            self.DEBUG("unravelled actions:", action)

        self.DEBUG("Active player", self.player_list[self.activePlayer])

        ## handle the action and move the game to the new state
        self.gameState = self.gameState.handle(action, self)
        assert issubclass(self.gameState, GameState)

        terminated = self.gameState == TerminatedState
        if not terminated:
            if self.checkStatus():
                self.gameState = RewardState

        ## if the number of steps has gone above maximum, we'll truncate the game here
        truncated = self._stepCount > self._maxSteps

        ## set this so that on the outside, we know which player we should give the reward to
        reward = self.player_list[self.activePlayer].claimReward()

        self._info["winner"] = self._winner
        self._info["reward"] = reward

        if not terminated:
            if truncated:
                self.INFO("::::: Game Truncated :::::")
                step = ts.truncation(
                    reward=reward,
                    discount=1.0,
                    observation={
                        "observation": self.getObservation(self.activePlayer),
                        "mask": self.get_mask(self.activePlayer),
                        "activePlayer": self.activePlayer,
                    },
                )

            else:
                step = ts.transition(
                    reward=reward,
                    discount=1.0,
                    observation={
                        "observation": self.getObservation(self.activePlayer),
                        "mask": self.get_mask(self.activePlayer),
                        "activePlayer": self.activePlayer,
                    },
                )

        else:
            self.INFO("::::: Game Terminated :::::")
            step = ts.termination(
                reward=reward,
                observation={
                    "observation": self.getObservation(self.activePlayer),
                    "mask": self.get_mask(self.activePlayer),
                    "activePlayer": self.activePlayer,
                },
            )

        self._stepCount += 1
        return step


class GameState(ABC):
    """Abstract base class representing a game state

    Users should derive states from this object when implementing a game.
    Will need to implement a handle() method which should perform actions specified by a player.
    """

    @abstractmethod
    def handle(action: np.ndarray, game: Game):
        """abstract method to step within the state, should advance the state of the Game object based on the provided action array."""


class RewardState(GameState):
    """Special game state which occurs at the end of play in which final rewards are handed out to players"""

    def handle(action: np.ndarray, game: Game) -> GameState:
        game.activePlayer = (game.activePlayer + 1) % game.nPlayers

        if game.activePlayer == 0:
            game.INFO("========= DONE HANDING OUT REWARDS ==========")
            return TerminatedState

        return RewardState


class TerminatedState(GameState):
    """Special game state used to indicate that the game has been terminated"""

    def handle(action: np.ndarray, game: Game) -> GameState:
        raise Exception(
            "handle() should never actually get called for TerminatedState. Something has gone wrong!!"
        )
