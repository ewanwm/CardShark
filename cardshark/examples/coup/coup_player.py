"""Define the Player object for use in the Coup game"""

# Python stuff
import numpy as np

# CardShark stuff
from cardshark import engine


class CoupPlayer(engine.Player):
    """Player class for the Coup game"""

    def __init__(self, n_cards, **kwargs):
        self.n_cards = n_cards
        self.coins = None
        self.is_alive = None

        super().__init__(**kwargs)

    def reset(self):
        """Reset the player"""

        self.coins = 2  ## start with 2 coins
        self.cards = []
        self.card_states = []
        self.is_alive = True
        self._reward_accum = 0

    def give_card(self, card_name):
        """Give a card to this player

        Will raise an exception if adding the card would take the number
        of cards the player has over the specified limit
        """

        if len(self.cards) >= self.n_cards:
            raise ValueError(
                "ERROR: trying to give player "
                + self.name
                + " a card when they already have "
                + len(self.cards)
                + " cards. Max num of cards is set to "
                + str(self.n_cards)
            )

        self.cards.append(card_name)
        self.card_states.append("Alive")

    def kill(self):
        """Kill the player"""

        self.info("AAARGH, I'm dead!")
        self.is_alive = False
        self.give_reward(-30)

    def take_card(self, card_name):
        """Take a particular card away from this player

        Will check if the player actually has the specified card and
        it is alive and raise an exception if not.
        """

        for i in range(self.n_cards):
            if (self.card_states[i] == "Alive") & (self.cards[i] == card_name):
                self.cards.pop(i)
                self.card_states.pop(i)
                return

        raise RuntimeError(
            "ERROR: trying to take card",
            card_name,
            "away from player",
            self.name,
            "but they do not have one that is alive",
        )

    def check_card(self, card_name) -> bool:
        """Check if the player has a particular card

        return true if they have the card and it is alive or false otherwise
        """

        for i in range(self.n_cards):
            if (self.card_states[i] == "Alive") & (self.cards[i] == card_name):
                return True

        return False

    def give_coins(self, n_coins):
        """Give the player some coins"""

        self.coins += n_coins

    def take_coins(self, n_coins):
        """Take some coins away from the player

        Will raise an exception if trying to take away more coins than
        the player has
        """

        if self.coins - n_coins < 0:
            raise ValueError(
                "ERROR: trying to take "
                + str(n_coins)
                + " from player "
                + self.name
                + " Who only has "
                + str(self.coins)
            )
        self.coins = self.coins - n_coins

    def lose_influence(self):
        """Kill one of the players cards at random"""

        if np.all(np.array(self.card_states) == "Dead"):
            raise RuntimeError(
                "ERROR: trying to make",
                self.name,
                "lose influence but all their cards are already dead: ",
                self,
            )

        alive_cards = np.where(np.array(self.card_states) == "Alive")[0]
        card_idx = np.random.choice(alive_cards)

        self.info("Losing influence. Card: ", self.cards[card_idx])
        self.card_states[card_idx] = "Dead"

        self.give_reward(-10)

    def info_string(self):
        """Get information string for debugging"""

        ret_str = "N coins: " + str(self.coins) + ", "
        ret_str += "N cards: " + str(len(self.cards)) + ", "
        ret_str += "cards: {"
        for i, card in enumerate(self.cards):
            ret_str += card + ":" + self.card_states[i][0] + ", "
        ret_str += "}"

    def observe(
        self,
        colour: str,
        dead_card_colour: str,
        reset_colour: str,
        coin_colour: str,
        full=False,
    ):
        """Get the "observation" of this player

        i.e. what another player should see when observing this one.
        full: whether to print the "full" information about this player.
            If true then print all cards held by the player.
        """

        card_string = "["
        for card, state in zip(self.cards, self.card_states):
            card_name = ""
            if full:
                card_name = card
            else:
                card_name = "unknown"

            if state == "Dead":
                card_string += dead_card_colour + f"{card:>10}" + colour + ", "
            else:
                card_string += f"{card_name:>10}" + ", "

        card_string += "] "

        ret_str = ":: {col}{name}{col} has {coin_col}{n_coins} coins{col} and cards " \
        "{cards}{reset}::".format(
            name=f"{self.name:>15}",
            n_coins=self.coins,
            cards=card_string,
            col=colour,
            reset=reset_colour,
            coin_col=coin_colour,
        )
        return ret_str
