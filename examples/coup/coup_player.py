
from cardshark import player
import numpy as np

class CoupPlayer(player.Player):
    def __init__(self, nCards, **kwargs):
        
        self.nCards = nCards

        super().__init__(**kwargs)

    def reset(self):
        self.Coins = 2 ## start with 2 coins
        self.Cards = []
        self.CardStates = []
        self.isAlive = True
        self.rewardAccum = 0
        
    def giveCard(self, cardName):

        if len(self.Cards) >= self.nCards:
            raise Exception("ERROR: trying to give player " + self.name + " a card when they already have " + len(self.Cards) + " cards. Max num of cards is set to " + str(self.nCards))

        self.Cards.append(cardName)
        self.CardStates.append("Alive")

    def kill(self):
        self.INFO("AAARGH, I'm dead!")
        self.isAlive = False
        self.giveReward(-30)

    def takeCard(self, cardName):
        for i in range(self.nCards):
            if (self.CardStates[i] == "Alive") & (self.Cards[i] == cardName):
                self.Cards.pop(i)
                self.CardStates.pop(i)
                return
        
        raise Exception("ERROR: trying to take card", cardName, "away from player", self.name, "but they do not have one that is alive")

    def checkCard(self, cardName):
        for i in range(self.nCards):
            if (self.CardStates[i] == "Alive") & (self.Cards[i] == cardName):
                return True

    def giveCoins(self, nCoins):
        self.Coins += nCoins

    def takeCoins(self, nCoins):
        if self.Coins - nCoins < 0:
            raise Exception("ERROR: trying to take " + str(nCoins) + " from player " + self.name + " Who only has " + str(self.Coins))
        self.Coins = self.Coins - nCoins
    
    def loseInfluence(self, cardIdx):
        ## kill card with index cardIdx
        self.INFO("Losing influence. Card: ", self.Cards[cardIdx])
        self.cardStates[cardIdx] = "Dead"

        self.giveReward(-10)

    def loseInfluence(self):
        ## kill one of the players cards at random
        if np.all(np.array(self.CardStates) == "Dead"):
            raise Exception("ERROR: trying to make", self.name, "lose influence but all their cards are already dead: ", self.__str__())
        
        aliveCards = np.where(np.array(self.CardStates) == "Alive")[0]
        cardIdx = np.random.choice(aliveCards)

        self.INFO("Losing influence. Card: ", self.Cards[cardIdx])
        self.CardStates[cardIdx] = "Dead"
        
        self.giveReward(-10)

        
    def info_string(self):
        retStr = "N Coins: " + str(self.Coins) + ", "
        retStr += "N Cards: " + str(len(self.Cards)) + ", "
        retStr += "Cards: {"
        for i, card in enumerate(self.Cards):
            retStr += card + ":" + self.CardStates[i][0] + ", "
        retStr += "}"

    def observe(self, colour: str, dead_card_colour: str, reset_colour: str, coin_colour: str, full=False):
        """Print out the "observation" of this player. i.e. what another player should see when observing this one

        full: whether to print the "full" information about this player. If true then print all cards held by the player.
        """

        card_string = "["
        for card, state in zip(self.Cards, self.CardStates):
            card_name = ""
            if full:
                card_name = card
            else:
                card_name = "unknown"
            
            if state == "Dead":
                card_string += dead_card_colour + f'{card:>10}' + colour + ", "
            else:
                card_string += f'{card_name:>10}' + ", "

        card_string += "] "

        retStr = ":: {col}{name}{col} has {coin_col}{n_coins} coins{col} and cards {cards}{reset}::".format(name=f'{self.name:>15}', n_coins = self.Coins, cards=card_string, col=colour, reset=reset_colour, coin_col=coin_colour)
        return retStr