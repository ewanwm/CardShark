"""Define Agent for Coup game

Currently just a pretty simple command line interface.
"""

# Python stuff
import numpy as np

# CardShark stuff
from cardshark.agent import HumanAgent
from cardshark.examples.coup import coup_engine


GRAY = "\u001b[30;1m"
RED = "\u001b[31;1m"
CYAN = "\u001b[36;1m"
YELLOW = "\u001b[33;1m"
GREEN = "\u001b[32;1m"
MAGENTA = "\u001b[35;1m"
RESET = "\u001b[0m"


class CoupHumanAgent(HumanAgent):
    """Human UI for Coup game
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._game = None
        self._player_id = None

    def _print_observations(self):
        print(
            "\n########################## Observations ##############################"
        )
        for i, player in enumerate(self._game.player_list):
            if i == self._player_id:
                print(
                    player.observe(
                        colour=GREEN,
                        dead_card_colour=RED,
                        reset_colour=RESET,
                        coin_colour=YELLOW,
                        full=True,
                    )
                    + " (YOU)"
                )
            else:
                print(
                    player.observe(
                        colour=CYAN,
                        dead_card_colour=RED,
                        reset_colour=RESET,
                        coin_colour=YELLOW,
                        full=False,
                    )
                )

        print(
            "######################################################################\n"
        )

    def _print_options(self, dim, mask=None):
        """Prints out the players options corresponding to the specified action array dimension"""

        ## set what to print depending on which action dimension was specified
        if dim == 0:
            print("Choose an action: ")
            names = coup_engine.ACTION_NAMES
        elif dim == 1:
            print("Choose another player to target: ")
            names = []
            for other_id in range(1, self._game.n_players):
                names.append(
                    self._game.player_list[
                        (other_id + self._player_id) % self._game.n_players
                    ].name
                )
        elif dim == 2:
            print("What would you like to do?")
            names = ["Nothing", "Challenge", "Block"]
        elif dim == 3:
            print("Do you want to try to challenge this block?")
            names = ["No", "Yes"]

        ## now print the options
        for i, name in enumerate(names):
            if mask is not None and mask[i] == 0:
                colour = GRAY
            else:
                colour = RESET

            print(
                "  {colour}{i}. {action}{colour_reset}".format(
                    colour=colour, i=i, action=name, colour_reset=RESET
                )
            )

    def _get_block_challenge_mask(self):
        n_targets = int(self._game._action_space.action_max[1])

        mask = np.zeros(3)

        for value in range(3):
            if self._game._get_mask(np.array([0, n_targets - 1, value, 0]), self._player_id):
                mask[value] = 1

        return mask

    def _get_action_mask(self):
        
        n_actions = int(self._game._action_space.action_max[0])
        mask = np.zeros(n_actions)
        for action in range(n_actions):
            ## check if this action is valid for at least one player
            for player_id in range(self._game.n_players):
                if self._game._get_mask(np.array([action, player_id, 0, 0, 0]), self._player_id):
                    mask[action] = 1
        
        return mask

    def _get_target_mask(self, action: int):

        n_targets = int(self._game._action_space.action_max[1])
        mask = np.zeros(n_targets)

        for target in range(n_targets):
            if self._game._get_mask(np.array([action, target, 0, 0, 0]), self._player_id):
                mask[target] = 1.0
        
        return mask

    def _get_user_input(self, max: int, mask=None) -> int:
        
            # get user input and check it's valid
            while True:
                inp = input("choice: ")
                valid = True
                message = ""

                if not inp.isdigit():
                    valid = False
                    message = "Must be an integer value"

                inp_int = int(inp)

                if inp_int >= max:
                    valid = False
                    message = "Value out of range"

                elif mask is not None:
                    if mask[inp_int] == 0:
                        valid = False
                        message = "Option not allowed"

                if valid:
                    return inp_int

                ## if not valid, tell the user why and loop will repeat
                print("{}INVALID CHOICE: {}{}".format(RED, message, RESET))

    def _get_action(self):
        ## 1st variable is which action to take in the action phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether to try to block current attemted action in blocking phase
        ## 4th variable is whether to challenge the acting player in the challenge phase
        ## 5th variable is whether to challenge the attempted block

        ret = np.zeros(4, dtype=int)

        if not self._game.player_list[self._player_id].is_alive:
            return ret
        
        self._print_observations()

        if self._game._game_state == coup_engine.ActionState:

            ## print possible actions and get desired one from player
            self._print_options(0, self._get_action_mask())
            ret[0] = self._get_user_input(self._game._action_space.action_max[0], self._get_action_mask())

            # check that at least one target is valid
            if np.sum(self._get_target_mask(ret[0])[:-1]) >= 1:
                ## print possible targets and get desired one from player
                self._print_options(1, self._get_target_mask(ret[0]))
                ret[1] = self._get_user_input(self._game._action_space.action_max[1], self._get_target_mask(ret[0]))
            else:
                ret[1] = self._game.n_players - 1

        elif self._game._game_state in [coup_engine.BlockingGeneralState, coup_engine.ChallengeGeneralState, coup_engine.BlockOrChallengeState]:

            mask = self._get_block_challenge_mask()
            self._print_options(2, mask)
            ret[2] = self._get_user_input(self._game._action_space.action_max[2], mask)
            
        elif self._game._game_state == coup_engine.ChallengeBlockState:

            self._print_options(3)
            ret[3] = self._get_user_input(self._game._action_space.action_max[3])


        return ret
