"""Derived engine classes to run Coup game
"""

# python stuff
from enum import Enum
import numpy as np

# TF stuff
# TODO: abstract all tf_agents out into Game base class. User shouldn't have to touch it
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

# cardshark engine stuff
from cardshark import engine
from cardshark.cards import Deck

# coup specific stuff
from examples.coup import coup_player
from examples.coup import coup_cards


## number of cards dealt to each player
MAX_CARDS = 2

## possible actions players can take
##         Action | Card Needed | Blocked by | Cost | isTargeted
actions = {
    "None": {
        "needs": "", 
        "blockedBy": [""], 
        "cost": 0, 
        "targeted": False
    },
    "income": {
        "needs": "", 
        "blockedBy": [""], 
        "cost": 0, 
        "targeted": False
    },
    "foreign_aid": {
        "needs": "", 
        "blockedBy": ["Duke"], 
        "cost": 0, 
        "targeted": False
    },
    "coup": {
        "needs": "", 
        "blockedBy": [""], 
        "cost": 7, 
        "targeted": True
    },
    "tax": {
        "needs": "Duke", 
        "blockedBy": [""], 
        "cost": 0, 
        "targeted": False
    },
    "steal": {
        "needs": "Captain",
        "blockedBy": ["Captain", "Inquisitor"],
        "cost": 0,
        "targeted": True,
    },
    "assassinate": {
        "needs": "Assassin",
        "blockedBy": ["Contessa"],
        "cost": 3,
        "targeted": True,
    },
    "exchange": {
        "needs": "Inquisitor",
        "blockedBy": [""],
        "cost": 0,
        "targeted": False,
    },
    "examine": {
        "needs": "Inquisitor", 
        "blockedBy": [""], 
        "cost": 0, 
        "targeted": True
    },
}


## used for converting action id to name
ACTION_NAMES = list(actions.keys())
ACTION_STRING = ""
for action_id, name in enumerate(ACTION_NAMES):
    if action_id != 0:
        ACTION_STRING = ACTION_STRING + " " + name
    else:
        ACTION_STRING = name
## used for converting action name to id
ActionEnum = Enum("ActionEnum", ACTION_STRING)


class CoupGame(engine.Game):
    """Class to represent Coup game
    """

    def __init__(
        self,
        n_players: int,
        unravel_action_space: bool = False,
        max_steps: int = np.inf,
        **kwargs,
    ):

        engine.Game.__init__(self, n_players, unravel_action_space, max_steps, **kwargs)

        ## make the action space spec
        ## 1st variable is which action to take in the action phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether to try to block current attemted action in blocking phase
        ## 4th variable is whether to challenge the acting player in the challenge phase
        ## 5th variable is whether to challenge the attempted block
        ### TODO: I think 3rd and 4th variable should be merged into one with
        ###   3 options: none, block, and challenge
        ###   so that situations where either block or challenge are valid can be
        ###   dealt with more easily in situations where only one is valid,
        ###   can easily just mask out the other one.
        action_spec_min = np.ndarray((5))
        action_spec_min[:] = 0
        action_spec_max = np.ndarray((5))
        action_spec_max[0] = len(actions.keys())
        action_spec_max[1] = self.n_players
        action_spec_max[2] = 2
        action_spec_max[3] = 2
        action_spec_max[4] = 2

        self.debug("action_spec_max: ", action_spec_max)
        self._action_spec = BoundedArraySpec(
            minimum=action_spec_min,
            maximum=action_spec_max,
            shape=action_spec_max.shape,
            dtype=np.int32,
        )  ##gym.spaces.MultiDiscrete(actionSpecNP)

        self._unravel_action_space = unravel_action_space
        if unravel_action_space:
            self._do_unravel_action_space()

        ## make array defining the observation spec
        ## this is the cards, and number of coins for each player
        obs_spec_max = np.ndarray(
            (self.n_players, 1 + MAX_CARDS), dtype=np.float32
        )
        obs_spec_min = np.ndarray(
            (self.n_players, 1 + MAX_CARDS), dtype=np.float32
        )
        obs_spec_min[:] = 0.0
        obs_spec_max[:, 0] = (
            12  # <- 12 is the max number of coins a player can have
        )
        obs_spec_max[:, 1:] = (
            len(coup_cards.cards.keys()) + 1
        )  # <- one index for each possible card + one for face down card

        self.debug("obs_spec_min: ", obs_spec_min)
        self.debug("obs_spec_max: ", obs_spec_max)

        self._observation_spec = {
            "observations": BoundedArraySpec(
                minimum=obs_spec_min.flatten(),
                maximum=obs_spec_max.flatten(),
                shape=obs_spec_max.flatten().shape,
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
                maximum=self.n_players,
                shape=(),
                dtype=np.int32,
                name="activePlayer",
            ),
        }

        ## TODO: Add an observation for which player is targetting this player with an action
        ## TODO: Add an observation for which action is currently being attempted and by who

        ## initialise the players
        ## TODO: Move this to Game initialiser. Need to have some register_player_class() method
        self.trace("  Creating players")
        for _ in range(n_players):
            self.player_list.append(
                coup_player.CoupPlayer(n_cards=MAX_CARDS, logger=self.logger)
            )

        ## initialise the deck
        self.trace("  Creating deck")
        self.deck = Deck(coup_cards.cards)

        self._reset()

    def action_array_to_string(self, action: np.array) -> str:
        """Convert action array to a human readable string

        Handy for debugging.
        """

        ret_str = "Action: {action}, Target: {target}, {block}, {challenge}, {challenge_block}"

        action_str = ACTION_NAMES[action[0]]

        if action[1] == self.n_players - 1:
            target_str = "None"
        else:
            target_str = action[1]

        block_str = "don't block" if (action[2] == 0) else "block"
        challenge_str = "don't challenge" if (action[3] == 0) else "challenge"
        challenge_block_str = (
            "don't challenge block" if (action[4] == 0) else "challenge block"
        )

        return ret_str.format(
            action=action_str,
            target=target_str,
            block=block_str,
            challenge=challenge_str,
            challenge_block=challenge_block_str,
        )

    def _reset(self):
        self.debug("Resetting game")
        ## set individual pieces back to initial state
        for player in self.player_list:
            player.reset()

        self.deck.reset()
        self.debug("Un shuffled deck:", self.deck)
        self.deck.shuffle()
        self.debug("shuffled deck:", self.deck)

        ## hand out the cards
        for _ in range(MAX_CARDS):
            for player in self.player_list:
                player.give_card(self.deck.draw())

        self.debug("deck after dealing:", self.deck)
        self.debug("Players:")
        for player in self.player_list:
            self.debug("  ", player)

        ## set initial game state
        self._game_state = ActionState
        self.attempted_action = "NOT_SET"
        self.current_player_action = 0
        self.set_active_player(0)
        self.action_target = 999
        self.current_player_block = 999
        self.current_player_challenge = 999
        self._winner = -999

        self._step_count = 0
        self._info = {}

        self._info["reward"] = 0
        self._info["skippingTurn"] = False

        return ts.restart(
            observation={
                "observation": self.get_observation(self.get_active_player()),
                "mask": self.get_mask(self.get_active_player()),
                "activePlayer": self.get_active_player(),
            }
        )

    ############################################################################################
    #### These are the functions that actually perform the actions specified by the players ####
    ############################################################################################
    def _income(self, p):
        player = self.player_list[p]
        self.info("  - Player: " + player.name + " performed action: income (+1 coin)")
        player.give_coins(1)

        player.give_reward(1)

    def _foreign_aid(self, p):
        player = self.player_list[p]
        self.info(
            "  - Player: " + player.name + " performed action: foreign_aid (+2 coins)"
        )
        player.give_coins(2)

        player.give_reward(2)

    def _coup(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.info(
            "  - Player: "
            + player1.name
            + " performed action: coup, Target: "
            + player2.name
        )
        player1.take_coins(actions["coup"]["cost"])
        player2.lose_influence()

        player1.give_reward(10)

    def _tax(self, p):
        player = self.player_list[p]
        self.info("  - Player: " + player.name + " performed action: tax (+3 coins)")
        player.give_coins(3)
        player.give_reward(3)

    def _steal(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]

        ## do this to avoid trying to steal 2 coins when target player doesn't have enoug
        ## this should probably be masked out but I'm not sure how to do that easily as it
        ## requires masking a specific combination of actions, (steal and specific players)
        ## BUT it's probably ok since the agents should learn to e.g. not steal from
        ## someone with 0 coins
        coins_to_steal = min(player2.coins, 2)

        self.info(
            "  - Player: "
            + player1.name
            + " performed action: steal, Target: "
            + player2.name
            + " (stole {})".format(coins_to_steal)
        )

        player2.take_coins(coins_to_steal)
        player1.give_coins(coins_to_steal)

        player2.give_reward(-coins_to_steal)
        player1.give_reward(coins_to_steal)

    def _assassinate(self, p1, p2):
        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.info(
            "  - Player: "
            + player1.name
            + " performed action: assassinate, Target: "
            + player2.name
        )
        player1.take_coins(3)
        player2.lose_influence()

        player1.give_reward(10)

    def _exchange(self):
        raise NotImplementedError("ERROR: exchange() not implemented yet")
        ## not yet implemented

    def _examine(self):
        raise NotImplementedError("ERROR: Examine() not implemented yet")
        ## not yet implemented

    def is_targetted_action(self, action: str):
        """Check if a particular action is targetted
        """
        return actions[action]["targeted"]

    def perform_attempted_action(self):
        """Perform the action currently stored in self.attempted_action
        """
        fn = getattr(
            self,
            "_" + self.attempted_action
        )  ## get the function corresponding to the attempted action

        if actions[self.attempted_action]["targeted"]:
            fn(self.current_player_action, self.action_target)
        else:
            fn(self.current_player_action)

    def _check_status(self):
        ## check the status of all the of all players
        ## for each one, check if all their cards are dead, if so, kill that player
        ## if all but one player is dead, they win and we're done
        ## returns true if game is finished, false otherwise

        ## first check each player
        for player in self.player_list:
            all_cards_dead = True
            for card_id in range(MAX_CARDS):
                if player.card_states[card_id] == "Alive":
                    all_cards_dead = False

            if all_cards_dead & player.is_alive:
                player.kill()

        alive_count = 0
        for player in self.player_list:
            if player.is_alive:
                alive_count += 1

        if alive_count == 1:
            self.info("=========== GAME OVER ===========")

            for player_id, player in enumerate(self.player_list):
                if player.is_alive:
                    self.info("")
                    self.info("  ** Player " + player.name + " Wins! **")
                    self._winner = player_id
                    player.give_reward(50)

                    return True

        return False

    def get_mask_ndim(self, player_id: int):
        """Get n dimensional mask
        
        Has same shape as the n dimensional action array.
        """

        ## make the action space spec
        ## 1st variable is which action to take in the action phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether to try to block current attemted action in blocking phase
        ## 4th variable is whether to challenge the acting player in the challenge phase
        ## 5th variable is whether to challenge the attempted block
        mask_list = []
        mask_list.append(np.ndarray((len(actions.keys())), dtype=np.int8))
        mask_list.append(np.ndarray((self.n_players,), dtype=np.int8))
        mask_list.append(np.ndarray((2,), dtype=np.int8))
        mask_list.append(np.ndarray((2,), dtype=np.int8))
        mask_list.append(np.ndarray((2,), dtype=np.int8))

        if self._game_state == ActionState:
            if self.player_list[player_id].coins < 10:  ##10:
                ## if player has >= 10 they can only perform a coup
                ## and their only choice is which player to target
                ## so we leave all mask values for actions at their default value of 0
                for action_id, action in enumerate(actions.keys()):
                    if self.player_list[player_id].coins >= actions[action]["cost"]:
                        mask_list[0][action_id] = 1
                    else:
                        mask_list[0][action_id] = 0

                    ## currently unimplemented actions
                    if action in ["exchange", "examine"]:
                        mask_list[0][action_id] = 0

                    ## should never actually get selected in action phase
                    if action == "None":
                        mask_list[0][action_id] = 0

            ## have >= 10 coins so can only perform coup
            else:
                self.debug("Player has >= 10 coins so can only perform coup")
                mask_list[0][:] = 0
                mask_list[0][ActionEnum["coup"].value - 1] = 1

            ## can only target players that are alive
            for other_player in range(1, self.n_players):
                if self.player_list[(other_player + player_id) % self.n_players].is_alive:
                    mask_list[1][other_player - 1] = 1
                else:
                    mask_list[1][other_player - 1] = 0

            mask_list[0][ActionEnum["None"].value - 1] = 0  ## don't allow no action
            mask_list[1][self.n_players - 1] = 0  ## don't allow no target

        else:
            mask_list[0][:] = 0
            mask_list[1][:] = 0

            mask_list[0][ActionEnum["None"].value - 1] = 1  ## allow no action
            mask_list[1][self.n_players - 1] = 1  ## allow no target

        mask_list[2][0] = 1
        if self._game_state in [BlockingGeneralState, BlockingTargetState]:
            mask_list[2][1] = 1
        else:
            mask_list[2][1] = 0

        mask_list[3][0] = 1
        if self._game_state in [ChallengeGeneralState, ChallengeTargetState]:
            mask_list[3][1] = 1
        else:
            mask_list[3][1] = 0

        mask_list[4][0] = 1
        if self._game_state == ChallengeBlockState:
            mask_list[4][1] = 1
        else:
            mask_list[4][1] = 0

        ## If in final reward state, player cant do anything
        if self._game_state == engine.RewardState:
            for mask_i in range(len(mask_list)):
                mask_list[mask_i][:] = 0

        ## If dead, player cant do anything
        if not self.player_list[player_id].is_alive:
            for mask_i in range(len(mask_list)):
                mask_list[mask_i][:] = 0

        self.debug("Mask: ", mask_list)

        return mask_list

    def get_mask(self, player_id: int) -> np.array:
        ## get action space mask for player at index player_id in this games player list
        self.debug(
            "Getting action mask for player",
            self.player_list[player_id].name,
            "at index",
            player_id,
        )

        mask_list = self.get_mask_ndim(player_id)

        if self._unravel_action_space:
            self.debug("Unravelling mask")
            unravelled_mask_list = []

            self.debug("  Allowed actions after unravelling: ")
            ## Now need to unravel the mask using the flattenedActionSpace found before
            for action in self._unravelled_action_space:
                all_allowed = 1
                for action_id, sub_action in enumerate(action):
                    all_allowed *= mask_list[action_id][sub_action]

                if all_allowed:
                    self.debug("   ", self.action_array_to_string(action))
                unravelled_mask_list.append(all_allowed)

            mask = np.array(unravelled_mask_list)

        return mask

    def get_observation(self, player_id: int) -> np.ndarray:
        self.debug(
            "Getting observation for player",
            self.player_list[player_id].name,
            "at index",
            player_id,
        )
        ## get observarion for player at index player_id in this games player list
        observation = np.ndarray((self.n_players, 1 + MAX_CARDS), dtype=np.float32)

        ## first fill in the observation for this player, always the first row in the observation
        ## so that any player will always see itself at the first position
        self.trace("Adding this players info")
        observation[0, 0] = self.player_list[player_id].coins
        ## can always see own cards
        for card_i in range(MAX_CARDS):
            observation[0, 1 + card_i] = coup_cards.CardEnum[
                self.player_list[player_id].cards[card_i]
            ].value

        ## for the rest of the observation we fill up the equivalent for other players
        for other_player_counter in range(1, self.n_players):
            other_player_id = (player_id + other_player_counter) % self.n_players
            self.trace(
                "adding info from player",
                self.player_list[other_player_id].name,
                "at index",
                other_player_id,
            )
            observation[other_player_counter, 0] = self.player_list[other_player_id].coins

            ## can only see other players cards if they are dead
            for card_id in range(MAX_CARDS):
                if self.player_list[other_player_id].card_states[card_id] == "Dead":
                    observation[other_player_counter, 1 + card_id] = coup_cards.CardEnum[
                        self.player_list[other_player_id].cards[card_id]
                    ].value
                else:
                    observation[other_player_counter, 1 + card_id] = 0.0

        self.debug("Observation Array:", observation)
        return observation.flatten()

    def swap_card(self, p, card):
        """Take card from player, return to deck, shuffle then draw a new one        
        """

        player = self.player_list[p]
        player.take_card(card)
        self.info("Player", player.name, "Swapping card", card)
        self.deck.add_card(card)
        self.deck.shuffle()
        new_card = self.deck.draw()
        player.give_card(new_card)
        self.debug("       was swapped for", new_card)

    def challenge(self, p1, p2, *cards):
        """Perform challenge

        p1 is challenging p2 on whether they have one of the specified cards.
        """

        player1, player2 = self.player_list[p1], self.player_list[p2]
        self.debug(
            "Player",
            player1.name,
            "challenging Player",
            player2.name,
            "on having one of",
            *cards,
        )

        ## first shuffle order of the cards just to be extra fair
        card_list = [*cards]
        np.random.shuffle(card_list)
        for card in card_list:
            if player2.check_card(card):
                self.info("Challenge failed,", player2.name, "had a", card)
                ## according to rules, player needs to return the card and get a new one
                self.swap_card(p2, card)
                player1.lose_influence()
                return "failed"

        ## if made it to this point, player2 didnt have any of the specified cards
        self.info("Challenge succeeded,", player2.name, "did not have any of", *cards)
        player2.lose_influence()

        return "succeeded"


class ActionState(engine.GameState):
    """State to handle general action state
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.current_player_action].is_alive:
            game.info(
                game.player_list[game.current_player_action].name,
                "is dead! skipping their action",
            )
            game._info["skippingTurn"] = True
            game.current_player_action = (game.current_player_action + 1) % game.n_players
            game.set_active_player(game.current_player_action)

            return ActionState

        else:
            game.attempted_action = ACTION_NAMES[action[0]]
            game.info(
                "Player",
                game.player_list[game.current_player_action].name,
                "is attempting action",
                game.attempted_action,
            )

            blockable = False
            targetted = False
            challengable = False
            ## first check if this action has a target
            if actions[game.attempted_action]["targeted"]:
                game.action_target = (
                    game.current_player_action + 1 + action[1]
                ) % game.n_players
                game.info(
                    "Targetting player",
                    game.player_list[game.action_target].name,
                    "at index",
                    game.action_target,
                )
                targetted = True
            else:
                game.debug("Not a targetted action")

            ## check if this action can be blocked by any card
            if actions[game.attempted_action]["blockedBy"] != [""]:
                game.debug(
                    "Could be blocked by", actions[game.attempted_action]["blockedBy"]
                )
                blockable = True
            else:
                game.debug("Not blockable")

            ## check if this action could be challenged
            if actions[game.attempted_action]["needs"] != "":
                game.debug(
                    "can be challenged as action requires",
                    actions[game.attempted_action]["needs"],
                )
                challengable = True
            else:
                game.debug("cant be challenged")

            if blockable:
                if targetted:
                    game.current_player_block = game.action_target
                    game.set_active_player(game.current_player_block)

                    return BlockingTargetState
                else:
                    game.current_player_block = (
                        game.current_player_action + 1
                    ) % game.n_players
                    game.set_active_player(game.current_player_block)

                    return BlockingGeneralState

            elif challengable:
                if targetted:
                    game.current_player_challenge = game.action_target
                    game.set_active_player(game.current_player_challenge)
                    return ChallengeTargetState

                else:
                    game.current_player_challenge = (
                        game.current_player_action + 1
                    ) % game.n_players
                    game.set_active_player(game.current_player_challenge)

                    return ChallengeGeneralState

            else:
                game.perform_attempted_action()
                game.current_player_action = (
                    game.current_player_action + 1
                ) % game.n_players
                game.set_active_player(game.current_player_action)

                return ActionState


class BlockingGeneralState(engine.GameState):
    """State to handle situation where any player can try to block an action
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.current_player_block].is_alive:
            game.info(
                game.player_list[game.current_player_block].name,
                "is dead so they can't really block anything",
            )
            game._info["skippingTurn"] = True
            game.current_player_block = (game.current_player_block + 1) % game.n_players
            game.set_active_player(game.current_player_block)

            return BlockingGeneralState

        else:
            if game.current_player_block == game.current_player_action:
                ## we have returned to the acting player, indicating that no one blocked the action
                game.info("action was not challenged by any player")
                game.perform_attempted_action()
                game.current_player_action = (
                    game.current_player_action + 1
                ) % game.n_players
                game.set_active_player(game.current_player_action)

                return ActionState

            else:
                if action[2] == 1:
                    game.info(
                        "player",
                        game.player_list[game.current_player_block].name,
                        "is attempting to block current action,",
                        game.attempted_action,
                    )
                    game.set_active_player(game.current_player_action)
                    return ChallengeBlockState

                else:
                    ## dont change state, just move to next player and let them block if they want
                    game.info(
                        "player",
                        game.player_list[game.current_player_block].name,
                        "did not try to block current action,",
                        game.attempted_action,
                    )
                    game.current_player_block = (
                        game.current_player_block + 1
                    ) % game.n_players
                    game.set_active_player(game.current_player_block)

                    return BlockingGeneralState


class BlockingTargetState(engine.GameState):
    """State to handle situation where targetted player can try to block an action
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[2] == 1:
            game.info(
                "player",
                game.player_list[game.current_player_block].name,
                "is attempting to block current action,",
                game.attempted_action,
            )
            game.set_active_player(game.current_player_action)

            return ChallengeBlockState

        else:
            game.info(
                "action was not challenged by",
                game.player_list[game.current_player_block].name,
            )
            game.perform_attempted_action()
            game.current_player_action = (game.current_player_action + 1) % game.n_players
            game.set_active_player(game.current_player_action)

            return ActionState


class ChallengeGeneralState(engine.GameState):
    """State to handle situation where any player can challeng an action
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if not game.player_list[game.current_player_challenge].is_alive:
            game.info(
                game.player_list[game.current_player_challenge].name,
                "is dead so they can't really challenge anything",
            )
            game._info["skippingTurn"] = True
            game.current_player_challenge = (
                game.current_player_challenge + 1
            ) % game.n_players
            game.set_active_player(game.current_player_challenge)

            return ChallengeGeneralState

        if game.current_player_challenge == game.current_player_action:
            ## have returned back to acting player indicating no one blocked the action
            game.info("action was not challenged by any player")
            game.perform_attempted_action()
            game.current_player_action = (
                game.current_player_action + 1
            ) % game.n_players
            game.set_active_player(game.current_player_action)

            return ActionState

        else:
            if action[3] == 1:
                game.info(
                    "player",
                    game.player_list[game.current_player_challenge].name,
                    "is challenging",
                    game.player_list[game.current_player_action].name,
                    "on their action,",
                    game.attempted_action,
                )
                if (
                    game.challenge(
                        game.current_player_challenge,
                        game.current_player_action,
                        actions[game.attempted_action]["needs"],
                    )
                    == "failed"
                ):
                    game.perform_attempted_action()

                ## If a challenge happened, it will be resolved straight away
                ## so we move back to the action state
                game.current_player_action = (
                    game.current_player_action + 1
                ) % game.n_players
                game.set_active_player(game.current_player_action)

                return ActionState

            else:
                game.info(
                    "player",
                    game.player_list[game.current_player_challenge].name,
                    "did not attempt to challenge",
                )
                ## dont change state, just move to next player and let them block if they want
                game.current_player_challenge = (
                    game.current_player_challenge + 1
                ) % game.n_players
                game.set_active_player(game.current_player_challenge)

                return ChallengeGeneralState


class ChallengeTargetState(engine.GameState):
    """State to handle situation where a targetted player can challenge an action
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[3] == 1:
            game.info(
                "player",
                game.player_list[game.current_player_challenge].name,
                "is attempting to challenge current action,",
                game.attempted_action,
            )
            if (
                game.challenge(
                    game.current_player_challenge,
                    game.current_player_action,
                    actions[game.attempted_action]["needs"],
                )
                == "failed"
            ):
                game.perform_attempted_action()

        else:
            game.info(
                "action was not challenged by",
                game.player_list[game.current_player_challenge].name,
            )
            game.perform_attempted_action()

        game.current_player_action = (game.current_player_action + 1) % game.n_players
        game.set_active_player(game.current_player_action)

        return ActionState


class ChallengeBlockState(engine.GameState):
    """State for handling player challenging an attempted block
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        if action[4] == 1:
            game.info(
                "player",
                game.player_list[game.current_player_action].name,
                "is challenging the attempted block by",
                game.player_list[game.current_player_block].name,
                "to block their action,",
                game.attempted_action,
            )
            game.challenge(
                game.current_player_action,
                game.current_player_block,
                *actions[game.attempted_action]["blockedBy"],
            )
        else:
            game.info(
                "player",
                game.player_list[game.current_player_action].name,
                "accepts attempted block by",
                game.player_list[game.current_player_block].name,
                "to block their action,",
                game.attempted_action,
            )
            game.player_list[game.current_player_action].give_reward(-3)
            game.player_list[game.current_player_block].give_reward(3)

        game.current_player_action = (game.current_player_action + 1) % game.n_players
        game.set_active_player(game.current_player_action)

        return ActionState


class BlockOrChallengeState(engine.GameState):
    """TODO: Implement this for when the target of an action can either block or challenge
    """

    @staticmethod
    def handle(action: np.ndarray, game: CoupGame) -> engine.GameState:
        raise NotImplementedError()
