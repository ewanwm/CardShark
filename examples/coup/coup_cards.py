from enum import Enum
from cardshark import logging as log

## define the initial deck configurations
##       Name   |  Initial number
cards = {"Duke": 4, "Captain": 4, "Assassin": 4, "Contessa": 4, "Inquisitor": 4}

## used for converting card id to name
card_names = list(cards.keys())
cardString = ""
for i, name in enumerate(card_names):
    if i != 0:
        cardString = cardString + " " + name
    else:
        cardString = name
## used for converting card name to id
cardEnum = Enum("cardEnum", cardString)
log.debug("cardString:", cardString)
