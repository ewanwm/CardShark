# cardshark engine stuff
from cardshark import engine
from cardshark.cards import Deck
from cardshark.logging import *

# python stuff
from enum import Enum
import numpy as np

# coup specific stuff
from examples.coup import coup_player
from examples.coup import coup_cards

# TF stuff
# TODO: abstract all tf_agents out into Game base class. User shouldn't have to touch it
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

## number of cards dealt to each player
MAX_CARDS = 2

## possible actions players can take
##         Action        | Card Needed          | Blocked by                           | Cost      | isTargeted
actions = {
    "None": {"needs": "", "blockedBy": [""], "cost": 0, "targeted": False},
    "Income": {"needs": "", "blockedBy": [""], "cost": 0, "targeted": False},
    "ForeignAid": {"needs": "", "blockedBy": ["Duke"], "cost": 0, "targeted": False},
    "Coup": {"needs": "", "blockedBy": [""], "cost": 7, "targeted": True},
    "Tax": {"needs": "Duke", "blockedBy": [""], "cost": 0, "targeted": False},
    "Steal": {
        "needs": "Captain",
        "blockedBy": ["Captain", "Inquisitor"],
        "cost": 0,
        "targeted": True,
    },
    "Assasinate": {
        "needs": "Assassin",
        "blockedBy": ["Contessa"],
        "cost": 3,
        "targeted": True,
    },
    "Exchange": {
        "needs": "Inquisitor",
        "blockedBy": [""],
        "cost": 0,
        "targeted": False,
    },
    "Examine": {"needs": "Inquisitor", "blockedBy": [""], "cost": 0, "targeted": True},
}


## used for converting action id to name
action_names = list(actions.keys())
action_string = ""
for i, name in enumerate(action_names):
    if i != 0:
        action_string = action_string + " " + name
    else:
        action_string = name
## used for converting action name to id
actionEnum = Enum("actionEnum", action_string)
DEBUG("action_string:", action_string)


class CoupGame(engine.Game):
    def __init__(
        self,
        nPlayers: int,
        unravelActionSpace: bool = False,
        maxSteps: int = np.inf,
        **kwargs,
    ):
        engine.Game.__init__(self, nPlayers, unravelActionSpace, maxSteps, **kwargs)

        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        ### TODO: I think 3rd and 4th variable should be merged into one with 3 options: none, block, and challenge
        ###  so that situations where either block or challenge are valid can be dealt with more easily
        ###  in situations where only one is valid, can easily just mask out the other one.
        actionSpecMin_NP = np.ndarray((5))
        actionSpecMin_NP[:] = 0
        actionSpecMax_NP = np.ndarray((5))
        actionSpecMax_NP[0] = len(actions.keys())
        actionSpecMax_NP[1] = self.nPlayers
        actionSpecMax_NP[2] = 2
        actionSpecMax_NP[3] = 2
        actionSpecMax_NP[4] = 2

        self.DEBUG("actionSpecMax_NP: ", actionSpecMax_NP)
        self._action_spec = BoundedArraySpec(
            minimum=actionSpecMin_NP,
            maximum=actionSpecMax_NP,
            shape=actionSpecMax_NP.shape,
            dtype=np.int32,
        )  ##gym.spaces.MultiDiscrete(actionSpecNP)

        self._unravelActionSpace = unravelActionSpace
        if unravelActionSpace:
            self._unravel_action_space()

        ## make array defining the observation spec
        ## this is the cards, and number of coins for each player
        observationSpecMax_NP = np.ndarray(
            (self.nPlayers, 1 + MAX_CARDS), dtype=np.float32
        )
        observationSpecMin_NP = np.ndarray(
            (self.nPlayers, 1 + MAX_CARDS), dtype=np.float32
        )
        observationSpecMin_NP[:] = 0.0
        observationSpecMax_NP[:, 0] = (
            12  # <- 12 is the max number of coins a player can have
        )
        observationSpecMax_NP[:, 1:] = (
            len(coup_cards.cards.keys()) + 1
        )  # <- one index for each possible card + one for face down card

        self.DEBUG("observationSpecMin_NP: ", observationSpecMin_NP)
        self.DEBUG("observationSpecMax_NP: ", observationSpecMax_NP)

        self._observation_spec = {
            "observations": BoundedArraySpec(
                minimum=observationSpecMin_NP.flatten(),
                maximum=observationSpecMax_NP.flatten(),
                shape=observationSpecMax_NP.flatten().shape,
                name="observation",
                dtype=np.float32,
            ),
            "mask": BoundedArraySpec(
                minimum=0,
                maximum=1,
                shape=(self._action_spec.maximum - self._action_spec.minimum + 1,),
                dtype=np.int32,
                name="mask",
            ),
            "activePlayer": BoundedArraySpec(
                minimum=0,
                maximum=self.nPlayers,
                shape=(),
                dtype=np.int32,
                name="activePlayer",
            ),
        }

        ## TODO: Add an observation for which player is targetting this player with an action
        ## TODO: Add an observation for which action is currently being attempted and by who

        ## initialise the players
        self.TRACE("  Creating players")
        self.player_list = []
        for _ in range(nPlayers):
            self.player_list.append(
                coup_player.CoupPlayer(nCards=MAX_CARDS, logger=self.logger)
            )

        ## initialise the deck
        self.TRACE("  Creating deck")
        self.Deck = Deck(coup_cards.cards)

        self._maxSteps = maxSteps
        self._reset()

    def action_array_to_string(self, action: np.array) -> str:
        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        retStr = "Action: {action}, Target: {target}, {block}, {challenge}, {challenge_block}"

        actionStr = action_names[action[0]]

        if action[1] == self.nPlayers - 1:
            targetStr = "None"
        else:
            targetStr = action[1]

        blockStr = "don't block" if (action[2] == 0) else "block"
        challengeStr = "don't challenge" if (action[3] == 0) else "challenge"
        challengeBlockStr = (
            "don't challenge block" if (action[4] == 0) else "challenge block"
        )

        return retStr.format(
            action=actionStr,
            target=targetStr,
            block=blockStr,
            challenge=challengeStr,
            challenge_block=challengeBlockStr,
        )

    def _reset(self):
        self.DEBUG("Resetting game")
        ## set individual pieces back to initial state
        for player in self.player_list:
            player.reset()

        self.Deck.reset()
        self.DEBUG("Un shuffled deck:", self.Deck)
        self.Deck.shuffle()
        self.DEBUG("shuffled deck:", self.Deck)

        ## hand out the cards
        for _ in range(MAX_CARDS):
            for player in self.player_list:
                player.giveCard(self.Deck.draw())

        self.DEBUG("deck after dealing:", self.Deck)
        self.DEBUG("Players:")
        for player in self.player_list:
            self.DEBUG("  ", player)

        ## set initial game state
        self.gameState = ActionState
        self.attemptedAction = "NOT_SET"
        self.currentPlayer_action = 0
        self.activePlayer = 0
        self.action_target = 999
        self.currentPlayer_block = 999
        self.currentPlayer_challenge = 999
        self._winner = -999

        self._stepCount = 0
        self._info = {}

        self._info["reward"] = 0
        self._info["skippingTurn"] = False

        return ts.restart(
            observation={
                "observation": self.getObservation(self.activePlayer),
                "mask": self.get_mask(self.activePlayer),
                "activePlayer": self.activePlayer,
            }
        )

    ############################################################################################
    #### These are the functions that actually perform the actions specified by the players ####
    ############################################################################################
    def _Income(self, p):
        player = self.player_list[p]
        self.INFO("  - Player: " + player.name + " performed action: Income (+1 coin)")
        player.giveCoins(1)

        player.giveReward(1)

    def _ForeignAid(self, p):
        player = self.player_list[p]
        self.INFO(
            "  - Player: " + player.name + " performed action: ForeignAid (+2 coins)"
        )
        player.giveCoins(2)

        player.giveReward(2)

    def _Coup(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.INFO(
            "  - Player: "
            + player1.name
            + " performed action: Coup, Target: "
            + player2.name
        )
        player1.takeCoins(actions["Coup"]["cost"])
        player2.loseInfluence()

        player1.giveReward(10)

    def _Tax(self, p):
        player = self.player_list[p]
        self.INFO("  - Player: " + player.name + " performed action: Tax (+3 coins)")
        player.giveCoins(3)
        player.giveReward(3)

    def _Steal(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]

        ## do this to avoid trying to steal 2 coins when target player doesn't have enoug
        ## this should probably be masked out but I'm not sure how to do that easily as it requires masking a specific combination of actions, (steal and specific players)
        ## BUT it's probably ok since the agents should learn to e.g. not steal from someone with 0 coins
        coinsToSteal = min(player2.Coins, 2)

        self.INFO(
            "  - Player: "
            + player1.name
            + " performed action: Steal, Target: "
            + player2.name
            + " (stole {})".format(coinsToSteal)
        )

        player2.takeCoins(coinsToSteal)
        player1.giveCoins(coinsToSteal)

        player2.giveReward(-coinsToSteal)
        player1.giveReward(coinsToSteal)

    def _Assasinate(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.INFO(
            "  - Player: "
            + player1.name
            + " performed action: Assassinate, Target: "
            + player2.name
        )
        player1.takeCoins(3)
        player2.loseInfluence()

        player1.giveReward(10)

    def _Exchange():
        raise Exception("ERROR: Exchange() not implemented yet")
        ## not yet implemented

    def _Examine():
        raise Exception("ERROR: Exchange() not implemented yet")
        ## not yet implemented

    def is_targetted_action(self, action: str):
        return actions[action]["targeted"]

    def performAttemptedAction(self):
        fn = self.__getattribute__(
            "_" + self.attemptedAction
        )  ## get the function corresponding to the attempted action

        if actions[self.attemptedAction]["targeted"]:
            fn(self.currentPlayer_action, self.action_target)
        else:
            fn(self.currentPlayer_action)

    def _checkStatus(self):
        ## check the status of all the of all players
        ## for each one, check if all their cards are dead, if so, kill that player
        ## if all but one player is dead, they win and we're done
        ## returns true if game is finished, false otherwise

        ## first check each player
        for player in self.player_list:
            allCardsDead = True
            for cardId in range(MAX_CARDS):
                if player.CardStates[cardId] == "Alive":
                    allCardsDead = False

            if allCardsDead & player.isAlive:
                player.kill()

        aliveCount = 0
        for player in self.player_list:
            if player.isAlive:
                aliveCount += 1

        if aliveCount == 1:
            self.INFO("=========== GAME OVER ===========")

            for playerIdx, player in enumerate(self.player_list):
                if player.isAlive:
                    self.INFO("")
                    self.INFO("  ** Player " + player.name + " Wins! **")
                    self._winner = playerIdx
                    player.giveReward(50)

                    return True

        return False

    def get_mask_ndim(self, playerIdx: int):
        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        maskList = []
        maskList.append(np.ndarray((len(actions.keys())), dtype=np.int8))
        maskList.append(np.ndarray((self.nPlayers,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))

        if self.gameState == ActionState:
            if self.player_list[playerIdx].Coins < 10:  ##10:
                ## if player has >= 10 they can only perform a coup
                ## and their only choice is which player to target
                ## so we leave all mask values for actions at their default value of 0
                for i, action in enumerate(actions.keys()):
                    if self.player_list[playerIdx].Coins >= actions[action]["cost"]:
                        maskList[0][i] = 1
                    else:
                        maskList[0][i] = 0

                    ## currently unimplemented actions
                    if action in ["Exchange", "Examine"]:
                        maskList[0][i] = 0

                    ## should never actually get selected in action phase
                    if action == "None":
                        maskList[0][i] = 0

            ## have >= 10 coins so can only perform coup
            else:
                self.DEBUG("Player has >= 10 coins so can only perform coup")
                maskList[0][:] = 0
                maskList[0][actionEnum["Coup"].value - 1] = 1

            ## can only target players that are alive
            for id in range(1, self.nPlayers):
                if self.player_list[(id + playerIdx) % self.nPlayers].isAlive:
                    maskList[1][id - 1] = 1
                else:
                    maskList[1][id - 1] = 0

            maskList[0][actionEnum["None"].value - 1] = 0  ## don't allow no action
            maskList[1][self.nPlayers - 1] = 0  ## don't allow no target

        else:
            maskList[0][:] = 0
            maskList[1][:] = 0

            maskList[0][actionEnum["None"].value - 1] = 1  ## allow no action
            maskList[1][self.nPlayers - 1] = 1  ## allow no target

        maskList[2][0] = 1
        if self.gameState in [BlockingGeneralState, BlockingTargetState]:
            maskList[2][1] = 1
        else:
            maskList[2][1] = 0

        maskList[3][0] = 1
        if self.gameState in [ChallengeGeneralState, ChallengeTargetState]:
            maskList[3][1] = 1
        else:
            maskList[3][1] = 0

        maskList[4][0] = 1
        if self.gameState == ChallengeBlockState:
            maskList[4][1] = 1
        else:
            maskList[4][1] = 0

        ## If in final reward state, player cant do anything
        if self.gameState == engine.RewardState:
            for i in range(len(maskList)):
                maskList[i][:] = 0

        ## If dead, player cant do anything
        if not self.player_list[playerIdx].isAlive:
            for i in range(len(maskList)):
                maskList[i][:] = 0

        self.DEBUG("Mask: ", maskList)

        return maskList

    def get_mask(self, playerIdx: int) -> np.array:
        ## get action space mask for player at index playerIdx in this games player list
        self.DEBUG(
            "Getting action mask for player",
            self.player_list[playerIdx].name,
            "at index",
            playerIdx,
        )

        maskList = self.get_mask_ndim(playerIdx)

        if self._unravelActionSpace:
            self.DEBUG("Unravelling mask")
            unravelledMaskList = []

            self.DEBUG("  Allowed actions after unravelling: ")
            ## Now need to unravel the mask using the flattenedActionSpace found before
            for action in self._unravelled_action_space:
                allAllowed = 1
                for i, subAction in enumerate(action):
                    allAllowed *= maskList[i][subAction]

                if allAllowed:
                    self.DEBUG("   ", self.action_array_to_string(action))
                unravelledMaskList.append(allAllowed)

            mask = np.array(unravelledMaskList)

        return mask

    def getObservation(self, playerIdx: int) -> np.ndarray:
        self.DEBUG(
            "Getting observation for player",
            self.player_list[playerIdx].name,
            "at index",
            playerIdx,
        )
        ## get observarion for player at index playerIdx in this games player list
        observation = np.ndarray((self.nPlayers, 1 + MAX_CARDS), dtype=np.float32)

        ## first fill in the observation for this player, always the first row in the observation
        ## so that any player will always see itself at the first position
        self.TRACE("Adding this players info")
        observation[0, 0] = self.player_list[playerIdx].Coins
        ## can always see own cards
        for i in range(MAX_CARDS):
            observation[0, 1 + i] = coup_cards.cardEnum[
                self.player_list[playerIdx].Cards[i]
            ].value

        ## for the rest of the observation we fill up the equivalent for other players
        for otherPlayerCounter in range(1, self.nPlayers):
            otherPlayerIdx = (playerIdx + otherPlayerCounter) % self.nPlayers
            self.TRACE(
                "adding info from player",
                self.player_list[otherPlayerIdx].name,
                "at index",
                otherPlayerIdx,
            )
            observation[otherPlayerCounter, 0] = self.player_list[otherPlayerIdx].Coins

            ## can only see other players cards if they are dead
            for i in range(MAX_CARDS):
                if self.player_list[otherPlayerIdx].CardStates[i] == "Dead":
                    observation[otherPlayerCounter, 1 + i] = coup_cards.cardEnum[
                        self.player_list[otherPlayerIdx].Cards[i]
                    ].value
                else:
                    observation[otherPlayerCounter, 1 + i] = 0.0

        self.DEBUG("Observation Array:", observation)
        return observation.flatten()

    def swapCard(self, p, card):
        ## take card from player, return to deck, shuffle then draw a new one
        player = self.player_list[p]
        player.takeCard(card)
        self.INFO("Player", player.name, "Swapping card", card)
        self.Deck.add_card(card)
        self.Deck.shuffle()
        newCard = self.Deck.draw()
        player.giveCard(newCard)
        self.DEBUG("       was swapped for", newCard)

    def challenge(self, p1, p2, *cards):
        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.DEBUG(
            "Player",
            player1.name,
            "challenging Player",
            player2.name,
            "on having one of",
            *cards,
        )

        ## first shuffle order of the cards just to be extra fair
        cardList = [*cards]
        np.random.shuffle(cardList)
        for card in cardList:
            if player2.checkCard(card):
                self.INFO("Challenge failed,", player2.name, "had a", card)
                ## according to rules, player needs to return the card and get a new one
                self.swapCard(p2, card)
                player1.loseInfluence()
                return "failed"

        ## if made it to this point, player2 didnt have any of the specified cards
        self.INFO("Challenge succeeded,", player2.name, "did not have any of", *cards)
        player2.loseInfluence()

        return "succeeded"


class ActionState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.currentPlayer_action].isAlive:
            game.INFO(
                game.player_list[game.currentPlayer_action].name,
                "is dead! skipping their action",
            )
            game._info["skippingTurn"] = True
            game.currentPlayer_action = (game.currentPlayer_action + 1) % game.nPlayers
            game.activePlayer = game.currentPlayer_action

            return ActionState

        else:
            game.attemptedAction = action_names[action[0]]
            game.INFO(
                "Player",
                game.player_list[game.currentPlayer_action].name,
                "is attempting action",
                game.attemptedAction,
            )

            blockable = False
            targetted = False
            challengable = False
            ## first check if this action has a target
            if actions[game.attemptedAction]["targeted"]:
                game.action_target = (
                    game.currentPlayer_action + 1 + action[1]
                ) % game.nPlayers
                game.INFO(
                    "Targetting player",
                    game.player_list[game.action_target].name,
                    "at index",
                    game.action_target,
                )
                targetted = True
            else:
                game.DEBUG("Not a targetted action")

            ## check if this action can be blocked by any card
            if actions[game.attemptedAction]["blockedBy"] != [""]:
                game.DEBUG(
                    "Could be blocked by", actions[game.attemptedAction]["blockedBy"]
                )
                blockable = True
            else:
                game.DEBUG("Not blockable")

            ## check if this action could be challenged
            if actions[game.attemptedAction]["needs"] != "":
                game.DEBUG(
                    "can be challenged as action requires",
                    actions[game.attemptedAction]["needs"],
                )
                challengable = True
            else:
                game.DEBUG("cant be challenged")

            if blockable:
                if targetted:
                    game.currentPlayer_block = game.action_target
                    game.activePlayer = game.currentPlayer_block

                    return BlockingTargetState
                else:
                    game.currentPlayer_block = (
                        game.currentPlayer_action + 1
                    ) % game.nPlayers
                    game.activePlayer = game.currentPlayer_block

                    return BlockingGeneralState

            elif challengable:
                if targetted:
                    game.currentPlayer_challenge = game.action_target
                    game.activePlayer = game.currentPlayer_challenge
                    return ChallengeTargetState

                else:
                    game.currentPlayer_challenge = (
                        game.currentPlayer_action + 1
                    ) % game.nPlayers
                    game.activePlayer = game.currentPlayer_challenge

                    return ChallengeGeneralState

            else:
                game.performAttemptedAction()
                game.currentPlayer_action = (
                    game.currentPlayer_action + 1
                ) % game.nPlayers
                game.activePlayer = game.currentPlayer_action

                return ActionState


class BlockingGeneralState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.currentPlayer_block].isAlive:
            game.INFO(
                game.player_list[game.currentPlayer_block].name,
                "is dead so they can't really block anything",
            )
            game._info["skippingTurn"] = True
            game.currentPlayer_block = (game.currentPlayer_block + 1) % game.nPlayers
            game.activePlayer = game.currentPlayer_block

            return BlockingGeneralState

        else:
            if game.currentPlayer_block == game.currentPlayer_action:
                ## we have returned to the acting player, indicating that no one blocked the action
                game.INFO("action was not challenged by any player")
                game.performAttemptedAction()
                game.currentPlayer_action = (
                    game.currentPlayer_action + 1
                ) % game.nPlayers
                game.activePlayer = game.currentPlayer_action

                return ActionState

            else:
                if action[2] == 1:
                    game.INFO(
                        "player",
                        game.player_list[game.currentPlayer_block].name,
                        "is attempting to block current action,",
                        game.attemptedAction,
                    )
                    game.activePlayer = game.currentPlayer_action
                    return ChallengeBlockState

                elif action[2] == 0:
                    ## we dont change state, just move to the next player and let them block if they want
                    game.INFO(
                        "player",
                        game.player_list[game.currentPlayer_block].name,
                        "did not try to block current action,",
                        game.attemptedAction,
                    )
                    game.currentPlayer_block = (
                        game.currentPlayer_block + 1
                    ) % game.nPlayers
                    game.activePlayer = game.currentPlayer_block

                    return BlockingGeneralState


class BlockingTargetState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[2] == 1:
            game.INFO(
                "player",
                game.player_list[game.currentPlayer_block].name,
                "is attempting to block current action,",
                game.attemptedAction,
            )
            game.activePlayer = game.currentPlayer_action

            return ChallengeBlockState

        else:
            game.INFO(
                "action was not challenged by",
                game.player_list[game.currentPlayer_block].name,
            )
            game.performAttemptedAction()
            game.currentPlayer_action = (game.currentPlayer_action + 1) % game.nPlayers
            game.activePlayer = game.currentPlayer_action

            return ActionState


class ChallengeGeneralState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.currentPlayer_challenge].isAlive:
            game.INFO(
                game.player_list[game.currentPlayer_challenge].name,
                "is dead so they can't really challenge anything",
            )
            game._info["skippingTurn"] = True
            game.currentPlayer_challenge = (
                game.currentPlayer_challenge + 1
            ) % game.nPlayers
            game.activePlayer = game.currentPlayer_challenge

            return ChallengeGeneralState

        else:
            if game.currentPlayer_challenge == game.currentPlayer_action:
                ## have returned back to acting player indicating no one blocked the action
                game.INFO("action was not challenged by any player")
                game.performAttemptedAction()
                game.currentPlayer_action = (
                    game.currentPlayer_action + 1
                ) % game.nPlayers
                game.activePlayer = game.currentPlayer_action

                return ActionState

            else:
                if action[3] == 1:
                    game.INFO(
                        "player",
                        game.player_list[game.currentPlayer_challenge].name,
                        "is challenging",
                        game.player_list[game.currentPlayer_action].name,
                        "on their action,",
                        game.attemptedAction,
                    )
                    if (
                        game.challenge(
                            game.currentPlayer_challenge,
                            game.currentPlayer_action,
                            actions[game.attemptedAction]["needs"],
                        )
                        == "failed"
                    ):
                        game.performAttemptedAction()

                    ## If a challenge happened, it will be resolved straight away so we move back to the action state
                    game.currentPlayer_action = (
                        game.currentPlayer_action + 1
                    ) % game.nPlayers
                    game.activePlayer = game.currentPlayer_action

                    return ActionState

                elif action[3] == 0:
                    game.INFO(
                        "player",
                        game.player_list[game.currentPlayer_challenge].name,
                        "did not attempt to challenge",
                    )
                    ## we dont change state, just move to the next player and let them block if they want
                    game.currentPlayer_challenge = (
                        game.currentPlayer_challenge + 1
                    ) % game.nPlayers
                    game.activePlayer = game.currentPlayer_challenge

                    return ChallengeGeneralState


class ChallengeTargetState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[3] == 1:
            game.INFO(
                "player",
                game.player_list[game.currentPlayer_challenge].name,
                "is attempting to challenge current action,",
                game.attemptedAction,
            )
            if (
                game.challenge(
                    game.currentPlayer_challenge,
                    game.currentPlayer_action,
                    actions[game.attemptedAction]["needs"],
                )
                == "failed"
            ):
                game.performAttemptedAction()

        else:
            game.INFO(
                "action was not challenged by",
                game.player_list[game.currentPlayer_challenge].name,
            )
            game.performAttemptedAction()

        game.currentPlayer_action = (game.currentPlayer_action + 1) % game.nPlayers
        game.activePlayer = game.currentPlayer_action

        return ActionState


class ChallengeBlockState(engine.GameState):
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[4] == 1:
            game.INFO(
                "player",
                game.player_list[game.currentPlayer_action].name,
                "is challenging the attempted block by",
                game.player_list[game.currentPlayer_block].name,
                "to block their action,",
                game.attemptedAction,
            )
            game.challenge(
                game.currentPlayer_action,
                game.currentPlayer_block,
                *actions[game.attemptedAction]["blockedBy"],
            )
        else:
            game.INFO(
                "player",
                game.player_list[game.currentPlayer_action].name,
                "accepts attempted block by",
                game.player_list[game.currentPlayer_block].name,
                "to block their action,",
                game.attemptedAction,
            )
            game.player_list[game.currentPlayer_action].giveReward(-3)
            game.player_list[game.currentPlayer_block].giveReward(3)

        game.currentPlayer_action = (game.currentPlayer_action + 1) % game.nPlayers
        game.activePlayer = game.currentPlayer_action

        return ActionState


class BlockOrChallengeState(engine.GameState):
    """TODO: Implement this for when the target of an action can either block or challenge the action"""

    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        raise NotImplementedError()
