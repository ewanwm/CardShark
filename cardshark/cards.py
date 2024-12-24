import cardshark
from cardshark.logging import *
import random
import typing
from abc import ABC, abstractmethod
from collections import Counter

class Deck:
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
        self._cards = list(self._init_cards)

    def shuffle(self):
        random.shuffle(self._cards)

    def draw(self):
        if(len(self._cards) <= 0):
            raise Exception("ERROR: Trying to draw from a deck with <= 0 cards in it")
        lastCard = self._cards[-1]
        self._cards.pop(-1)

        return lastCard
    
    def add_card(self, cardName):
        self._cards.append(cardName)

    def __str__(self):
        retStr = ""
        retStr += "Name: " + self.name + ", "
        retStr += "N Cards: " + str(len(self._cards)) + ", "

        counter = Counter(self._cards)

        retStr += "{"
        for card in counter.keys():
            retStr += card + ": " + str(self._cards.count(card)) + ", "
        retStr += "}"

        return retStr

class Card(ABC):
    """ABC representing a card

    This has basically no functionality or data and you're totally free to use it 
    however you want to... it feels a bit pointless tbh
    """