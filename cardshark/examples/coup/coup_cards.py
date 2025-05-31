"""Describe the cards used in Coup game"""

from enum import Enum
from cardshark import logging as log

## define the initial deck configurations
##       Name   |  Initial number
cards = {"Duke": 4, "Captain": 4, "Assassin": 4, "Contessa": 4, "Inquisitor": 4}

## used for converting card id to name
CARD_NAMES = list(cards.keys())
CARD_STRING = ""
for i, name in enumerate(CARD_NAMES):
    if i != 0:
        CARD_STRING = CARD_STRING + " " + name
    else:
        CARD_STRING = name

## used for converting card name to id
CardEnum = Enum("CardEnum", CARD_STRING)
log.debug("CARD_STRING:", CARD_STRING)
