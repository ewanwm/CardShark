from Logging import *
import random

## define the initial deck configurations
##       Name   |  Initial number
cards = {"Duke":       4,
         "Captain":    4,
         "Assassin":   4,
         "Contessa":   4,
         "Inquisitor": 4
}

## used for converting card id to name
cardNames  = list(cards.keys())
cardString = ""
for i, name in enumerate(cardNames): 
    if i!= 0:
        cardString = cardString + " " + name 
    else: cardString = name
## used for converting card name to id
cardEnum   = Enum("cardEnum", cardString)
DEBUG("cardString:", cardString)



class Deck:
    def __init__(self):
        self.name = "Deck"
        self.reset()

    def reset(self):
        self.Cards = []
        for cardName in cards.keys():
            for _ in range(cards[cardName]):
                self.Cards.append(cardName)

    def shuffle(self):
        random.shuffle(self.Cards)

    def draw(self):
        if(len(self.Cards) <= 0):
            raise Exception("ERROR: Trying to draw from a deck with <= 0 cards in it")
        lastCard = self.Cards[-1]
        self.Cards.pop(-1)

        return lastCard
    
    def returnCard(self, cardName):
        if not cardName in cards.keys():
            raise Exception("ERROR: Trying to return unknown card " + str(cardName) + " to deck")
        
        self.Cards.append(cardName)

    def __str__(self):
        retStr = ""
        retStr += "Name: " + self.name + ", "
        retStr += "N Cards: " + str(len(self.Cards)) + ", "

        retStr += "{"
        for cardName in cards.keys():
            retStr += cardName + ": " + str(self.Cards.count(cardName)) + ", "
        retStr += "}, "

        retStr += "["
        for card in self.Cards:
            retStr += card + ", "
        retStr += "]"

        return retStr
