from cardshark.agent import HumanAgent
from cardshark import logging as log

from examples.coup import coup_engine

import numpy as np

GRAY = "\u001b[30;1m"
RED = "\u001b[31;1m"
CYAN = "\u001b[36;1m"
YELLOW = "\u001b[33;1m"
GREEN = "\u001b[32;1m"
MAGENTA = "\u001b[35;1m"
RESET = "\u001b[0m"

class CoupHumanAgent(HumanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._game = None
        self._player_id = None

    def _print_observations(self):
        print("\n########################## Observations ##############################")
        for i, player in enumerate(self._game.player_list):
            if i == self._player_id:
                print(player.observe(colour=GREEN, dead_card_colour=RED, reset_colour=RESET, coin_colour=YELLOW, full=True) + " (YOU)")
            else:
                print(player.observe(colour=CYAN, dead_card_colour=RED, reset_colour=RESET, coin_colour=YELLOW, full=False))
                
        print("######################################################################\n")

    def _print_options(self, dim, mask_nd):
        """Prints out the players options corresponding to the specified action array dimension
        """

        ## set what to print depending on which action dimension was specified
        if dim == 0:
            print("Choose an action: ")
            names = coup_engine.action_names
        elif dim == 1:
            print("Choose another player to target: ")
            names = []
            for id in range(1, self._game.nPlayers):
                names.append(self._game.player_list[(id + self._player_id) % self._game.nPlayers].name)
        elif dim == 2:
            print("Do you want to try to block this action?")
            names = ["No", "Yes"]
        elif dim == 3:
            print("Do you want to try to challenge this action?")
            names = ["No", "Yes"]
        elif dim == 4:
            print("Do you want to try to challenge this block?")
            names = ["No", "Yes"]
            
        ## now print the options
        for i, name in enumerate(names):
            if mask_nd[dim][i] == 0:
                colour = GRAY
            else:
                colour = RESET

            print("  {colour}{i}. {action}{colour_reset}".format(colour=colour, i=i, action=name, colour_reset=RESET)) 

    def _get_action(self):
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase 
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block

        self._print_observations()

        mask_1d = self._game.get_mask(self._player_id)
        mask_nd = self._game.get_mask_ndim(self._player_id)

        ret = np.zeros(5, dtype=int)

        for dim in range(5):
            
            # if all options invalid skip this dim
            if( not np.any(np.array(mask_nd[dim]) == 1)):
                continue

            # if only one option valid then have to take it 
            if ( np.sum(np.array(mask_nd[dim])) == 1):
                ret[dim] = np.where(np.array(mask_nd[dim]) == 1)[0]
                continue

            ## horrible nasty ugly hack for when action is not targetted
            ## TODO: this really needs to be done inside the game and mask out player choice 
            ## when action is not targetted. This way ML agents will not need to worry about 
            ## trying to learn extra stuff when they don't need to
            if dim == 1:
                if(not self._game.is_targetted_action(coup_engine.action_names[ret[0]])):
                    ret[1] = self._game.nPlayers - 1
                    continue

            self._print_options(dim, mask_nd)

            # get user input and check it's valid
            while True:
                inp = input("choice: ")
                valid = True
                message = ""

                if not inp.isdigit():
                    valid = False
                    message = "Must be an integer value"
                
                inp_int = int(inp)
                
                if inp_int > len(mask_nd[dim]):
                    valid = False
                    message = "Value out of range"

                if mask_nd[dim][inp_int] == 0:
                    valid = False
                    message = "Option not allowed"

                if valid:
                    ret[dim] = inp_int
                    break

                ## if not valid, tell the user why and loop will repeat
                print("{}INVALID CHOICE: {}{}".format(RED, message, RESET))

        return ret
