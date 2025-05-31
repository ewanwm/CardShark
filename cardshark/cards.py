"""This module provides some useful functionality for general things related to cards

VERY lightweight at the moment but will likely flesh out a bit in the future.
"""

# Python stuff
import random
import typing
from collections import Counter

# CardShark stuff
from cardshark import logging as log


class Deck:
    """Class representing a deck of cards

    The actual objects it holds can be whatever you want. Can be simple strings,
    dictionaries or even whole classes, it's totally up to you!

    """

    def __init__(self, cards: typing.Union[list, dict], name="Deck"):
        self.name = name

        self._cards: list = []

        if isinstance(cards, list):
            self._cards = cards

        elif isinstance(cards, dict):
            for card in cards.keys():
                for _ in range(cards[card]):
                    self._cards.append(card)

        # keep a copy of the original state of the deck for resetting later
        self._init_cards = list(self._cards)

        self.reset()

    def reset(self):
        """Reset the deck

        Reset the internal list of card objects to what it was when the deck was instantiated
        """
        self._cards = list(self._init_cards)

    def shuffle(self):
        """Shuffle the order of the cards in the deck"""
        random.shuffle(self._cards)

    def draw(self):
        """Draw a card from the top of the deck

        Removes the card from the deck and returns it.
        If the deck has no more cards in it, will return None.
        """
        if len(self._cards) <= 0:
            log.error("Trying to draw from a deck with <= 0 cards in it")
            return None

        last_card = self._cards[-1]
        self._cards.pop(-1)

        return last_card

    def add_card(self, card_name):
        """Add a card to the top of this deck"""
        self._cards.append(card_name)

    def __str__(self):
        ret_str = ""
        ret_str += "Name: " + self.name + ", "
        ret_str += "N Cards: " + str(len(self._cards)) + ", "

        counter = Counter(self._cards)

        ret_str += "{"
        for card in counter.keys():
            ret_str += card + ": " + str(self._cards.count(card)) + ", "
        ret_str += "}"

        return ret_str
