import numpy as np
import PIL
from enum import Enum
import gym
import gym.spaces
import random
from Logging import *
#import tensorflow as tf
#print("Tensorflow version: ", tf.version.VERSION)
#import tf_agents as Agents

INFO("All modules loaded")

GAME_LOG_LEVEL = logLevels.kDebug

## number of cards dealt to each player
MAX_CARDS = 2

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

## possible actions players can take
##         Action        | Card Needed          | Blocked by                           | Cost      | isTargeted
actions = {"Income":      {"needs":"",           "blockedBy":[""],                      "cost":0,   "targeted":False},
           "Foreign Aid": {"needs":"",           "blockedBy":["Duke"],                  "cost":0,   "targeted":False},
           "Coup":        {"needs":"",           "blockedBy":[""],                      "cost":7,   "targeted":True},
           "Tax":         {"needs":"Duke",       "blockedBy":[""],                      "cost":0,   "targeted":False},
           "Steal":       {"needs":"Captain",    "blockedBy":["Captain", "Inquisitor"], "cost":0,   "targeted":True},
           "Assasinate":  {"needs":"Assassin",   "blockedBy":["Contessa"],              "cost":3,   "targeted":True},
           "Exchange":    {"needs":"Inquisitor", "blockedBy":[""],                      "cost":0,   "targeted":False},
           "Examine":     {"needs":"Inquisitor", "blockedBy":[""],                      "cost":0,   "targeted":True}
           }

## used for converting action id to name
actionNames  = list(actions.keys())
actionString = ""
for i, name in enumerate(actionNames): 
    if i != 0:
        actionString = actionString + " " + name 
    else: actionString = name
## used for converting action name to id
actionEnum   = Enum("actionEnum", actionString)
DEBUG("actionString:", actionString)

## possible states of play
gameStates = ["Action", "Blocking_general", "Blocking_target", "Challenge_general", "Challenge_target"]

class Deck:
    def __init__(self):
        self.Name = "Deck"
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
        retStr += "Name: " + self.Name + ", "
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


class Player:
    nPlayers = 0

    def __init__(self, playerName=""):
        if playerName == "":
            self.Name = "Player_" + str(Player.nPlayers)
            
        else: 
            self.Name = playerName

        self.Id = Player.nPlayers
        Player.nPlayers += 1

        self.reset()

    def reset(self):
        self.Coins = 2 ## start with 2 coins
        self.Cards = []
        self.CardStates = []

    def giveCard(self, cardName):
        if not cardName in cards.keys():
            raise Exception("ERROR: Trying to give unknown card " + str(cardName) + " to player " + self.Name)

        if len(self.Cards) >= MAX_CARDS:
            raise Exception("ERROR: trying to give player " + self.Name + " a card when they already have " + len(self.Cards) + " cards. Max num of cards is set to " + str(MAX_CARDS))

        self.Cards.append(cardName)
        self.CardStates.append("Alive")

    def takeCard(self, cardName):
        for i in range(MAX_CARDS):
            if (self.CardStates[i] == "Alive") & (self.Cards[i] == cardName):
                self.Cards.pop(i)
                return
        
        raise Exception("ERROR: trying to take card", cardName, "away from player", self.Name, "but they do not have one that is alive")

    def checkCard(self, cardName):
        for i in range(MAX_CARDS):
            if (self.CardStates[i] == "Alive") & (self.Cards[i] == cardName):
                return True

    def giveCoins(self, nCoins):
        self.Coins += nCoins

    def takeCoins(self, nCoins):
        if self.Coins - nCoins < 0:
            raise Exception("ERROR: trying to take " + str(nCoins) + " from player " + self.Name + " Who only has " + str(self.Coins))
        self.Coins = self.Coins - nCoins
    
    def loseInfluence(self, cardIdx):
        ## kill card with index cardIdx
        self.cardStates[cardIdx] = "Dead"

    def loseInfluence(self):
        ## kill one of the players cards at random
        if np.all(np.array(self.CardStates) == "Dead"):
            raise Exception("ERROR: trying to make", self.Name, "lose influence but all their cards are already dead: ", self.__str__())
        
        aliveCards = np.where(np.array(self.CardStates) == "Alive")[0]
        self.CardStates[np.random.choice(aliveCards)] = "Dead"
    
    def __str__(self):
        retStr = "Name: " + self.Name + ", "
        retStr += "N Coins: " + str(self.Coins) + ", "
        retStr += "N Cards: " + str(len(self.Cards)) + ", "
        retStr += "Cards: {"
        for i, card in enumerate(self.Cards):
            retStr += card + ":" + self.CardStates[i][0] + ", "
        retStr += "}"

        return retStr

class Game(gym.Env):
    nGames = 0
    def __init__(self, nPlayers, name=""):
        if name == "":
            self.name = "Game_" + str(Game.nGames)
        else:
            self.name = name
            
        DEBUG("Initialising Game:", self.name, ", with", nPlayers, "players")

        self.logger = Logger(GAME_LOG_LEVEL, self.name + "_Logger")

        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase 
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        actionSpecNP = np.ndarray((5))
        actionSpecNP[0] = len(actions.keys())
        actionSpecNP[1] = nPlayers-1
        actionSpecNP[2] = 2
        actionSpecNP[3] = 2
        actionSpecNP[4] = 2
        self.DEBUG("actionSpecNP: ", actionSpecNP)
        self.action_space = gym.spaces.MultiDiscrete(actionSpecNP)

        ## make array defining the observation spec
        ## this is the cards, and number of coins for each player
        observationSpecNP = np.ndarray((nPlayers, 1 + MAX_CARDS))
        observationSpecNP[:, 0] = 12 # <- 12 is the max number of coins a player can have
        observationSpecNP[:, 1:] = len(cards.keys()) +1 # <- one index for each possible card + one for face down card 
        self.DEBUG("observationSpecNP: ", observationSpecNP)
        self.observation_space = gym.spaces.MultiDiscrete(observationSpecNP)

        ## initialise the players
        self.TRACE("  Creating players")
        self.playerList = []
        self.nPlayers = nPlayers
        for _ in range(nPlayers):
            self.playerList.append(Player())

        ## initialise the deck
        self.TRACE("  Creating deck")
        self.Deck = Deck()

        Game.nGames += 1

        self.reset()

    def reset(self):
        self.DEBUG("Resetting game")
        ## set individual pieces back to initial state
        for player in self.playerList:
            player.reset()

        self.Deck.reset()
        self.DEBUG("Un shuffled deck:", self.Deck)
        self.Deck.shuffle()
        self.DEBUG("shuffled deck:", self.Deck)

        ## hand out the cards
        for _ in range(MAX_CARDS):
            for player in self.playerList:
                player.giveCard(self.Deck.draw())
        
        self.DEBUG("deck after dealing:", self.Deck)
        self.DEBUG("Players:")
        for player in self.playerList: self.DEBUG("  ", player)

        ## set initial game state
        self.gameState = "Action"
        self.attemptedAction = "NOT_SET"
        self.currentPlayer_action = 0
        self.action_target = 999
        self.currentPlayer_block = 999
        self.currentPlayer_challenge = 999

        return (self.getObservation(self.currentPlayer_action), {})

    def ERROR(self, *messages): self.logger.error(*messages)
    def WARN(self, *messages): self.logger.warn(*messages)
    def INFO(self, *messages): self.logger.info(*messages)
    def DEBUG(self, *messages): self.logger.debug(*messages)
    def TRACE(self, *messages): self.logger.trace(*messages)

    def Income(self, p):
        player = self.playerList[p]
        self.DEBUG(" Player: ", player.Name, "Action: Income")
        player.giveCoins(1)

    def ForeignAid(self, p):
        player = self.playerList[p]
        self.DEBUG(" Player: ", player.Name, "Action: ForeignAid")
        player.giveCoins(2)

    def Coup(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.DEBUG(" Player: ", player1.Name, "Action: Income, Target: ", player2.Name)
        player1.takeCoins(actions["Coup"]["cost"])
        player2.loseInfluence()

    def Tax(self, p):
        player = self.playerList[p]
        self.DEBUG(" Player: ", player.Name, "Action: Tax")
        player.giveCoins(3)

    def Steal(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.DEBUG(" Player: ", player1.Name, "Action: Steal, Target: ", player2.Name)
        player2.takeCoins(2)
        player1.giveCoins(2)

    def Assassinate(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.DEBUG(" Player: ", player1.Name, "Action: Assassinate, Target: ", player2.Name)
        player1.takeCoins(3)
        player2.loseInfluence()

    def Exchange():
        ## not yet implemented
        return

    def Examine():
        ## not yet implemented
        return
    
    def getMask(self, playerIdx):
        ## get action space mask for player at index playerIdx in this games player list
        self.DEBUG("Getting action mask for player", self.playerList[playerIdx].Name, "at index", playerIdx)
        
    
    def getObservation(self, playerIdx):
        self.DEBUG("Getting observation for player", self.playerList[playerIdx].Name, "at index", playerIdx)
        ## get observarion for player at index playerIdx in this games player list
        observation = np.ndarray((self.nPlayers, 1 + MAX_CARDS))

        ## first fill in the observation for this player, always the first row in the observation 
        ## so that any player will always see itself at the first position
        self.TRACE("Adding this players info")
        observation[0,0] = self.playerList[playerIdx].Coins
        ## can always see own cards
        for i in range(MAX_CARDS):
            observation[0, 1 + i] = cardEnum[self.playerList[playerIdx].Cards[i]].value

        ## for the rest of the observation we fill up the equivalent for other players
        for otherPlayerCounter in range(1, self.nPlayers):
            otherPlayerIdx = (playerIdx + otherPlayerCounter) % self.nPlayers
            self.TRACE("adding info from player", self.playerList[otherPlayerIdx].Name, "at index", otherPlayerIdx)
            observation[otherPlayerCounter, 0] = self.playerList[otherPlayerIdx].Coins

            ## can only see other players cards if they are dead                
            for i in range(MAX_CARDS):
                if self.playerList[otherPlayerIdx].CardStates[i] == "Dead":
                    observation[otherPlayerCounter, 1 + i] = cardEnum[self.playerList[otherPlayerIdx].Cards[i]].value     
                else:   
                    observation[otherPlayerCounter, 1 + i] = 0.0

        self.DEBUG("Observation Array:", observation)
        return observation
    
    def changeState(self, newState):
        if self.gameState == newState:
            self.WARN("Hmm, seem to be trying to move to state", newState, "but game is already in this state")

        self.DEBUG("moving from state", self.gameState, "to state", newState)
        self.gameState = newState

    def performAttemptedAction(self):
        if self.attemptedAction == "Income": fn = self.Income
        if self.attemptedAction == "Foreign Aid": fn = self.ForeignAid
        if self.attemptedAction == "Coup": fn = self.Coup
        if self.attemptedAction == "Tax": fn = self.Tax
        if self.attemptedAction == "Steal": fn = self.Steal
        if self.attemptedAction == "Assasinate": fn = self.Assassinate
        if self.attemptedAction == "Exchange": fn = self.Exchange
        if self.attemptedAction == "Examine": fn = self.Examine

        if actions[self.attemptedAction]["targeted"]:
            fn(self.currentPlayer_action, self.action_target)
        else:
            fn(self.currentPlayer_action)
    
    def swapCard(self, p, card):
        ## take card from player, return to deck, shuffle then draw a new one
        player=self.playerList[p]
        player.takeCard(card)
        self.Deck.returnCard(card)
        self.Deck.shuffle()
        player.giveCard(self.Deck.draw())

    def challenge(self, p1, p2, *cards):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.DEBUG("Player", player1.Name, "challenging Player", player2.Name, "on having one of", *cards)

        ## first shuffle order of the cards just to be extra fair
        cardList = [*cards]
        np.random.shuffle(cardList)
        for card in cardList:
            if player2.checkCard(card):
                self.DEBUG("Challenge failed,", player2.Name, "had a", card)
                ## according to rules, player needs to return the card and get a new one
                self.swapCard(p2, card)
                player1.loseInfluence()
                return "failed"

        ## if made it to this point, player2 didnt have any of the specified cards
        self.DEBUG("Challenge succeeded,", player2.Name, "did not have any of", *cards)
        player2.loseInfluence()

        return "succeeded"

    def step(self, action):
        self.DEBUG("stepping")
        self.DEBUG("gameState:",self.gameState)
        self.DEBUG("specified actions:", action)

        ## tings to return at the end of the step
        ret_observation = None
        ret_reward = 0
        ret_terminated = False
        ret_truncated = False
        ret_info = {}

        ## check what state we are in 

        ##### ACTION STATE #####
        if(self.gameState == "Action"):
            self.attemptedAction = actionNames[action[0]]
            self.DEBUG("Player", self.playerList[self.currentPlayer_action].Name, "is attempting action", self.attemptedAction)
            
            blockable = False
            targetted = False
            challengable = False
            ## first check if this action has a targed
            if actions[self.attemptedAction]["targeted"]:
                self.action_target = action[1]
                self.DEBUG("Targetting player", self.playerList[self.action_target].Name, "at index", self.action_target)
                targetted = True
            else:
                self.DEBUG("Not a targetted action")

            ## check if this action can be blocked by any card
            if actions[self.attemptedAction]["blockedBy"] != [""]:
                self.DEBUG("Could be blocked by", actions[self.attemptedAction]["blockedBy"])
                blockable = True
            else:
                self.DEBUG("Not blockable")

            ## check if this action could be challenged
            if actions[self.attemptedAction]["needs"] != "":
                self.DEBUG("can be challenged as action requires", actions[self.attemptedAction]["needs"])
                challengable = True
            else:
                self.DEBUG("cant be challenged")

            if(blockable):
                if targetted: 
                    self.currentPlayer_block = self.action_target
                    self.changeState("Blocking_target")
                else:
                    self.currentPlayer_block = (self.currentPlayer_action + 1) % self.nPlayers
                    self.changeState("Blocking_general")
                ret_observation = self.getObservation(self.currentPlayer_block)

            elif challengable:
                if targetted: 
                    self.currentPlayer_challenge = self.action_target
                    self.changeState("Challenge_target")
                else:
                    self.currentPlayer_challenge = (self.currentPlayer_action + 1) % self.nPlayers
                    self.changeState("Challenge_general")
                ret_observation = self.getObservation(self.currentPlayer_challenge)
        
            else:
                self.performAttemptedAction()
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                ret_observation = self.getObservation(self.currentPlayer_action)

        ##### GENERAL BLOCKING STATE #####
        elif(self.gameState == "Blocking_general"): ## state in which any player can attempt to block the attempted action
            if self.currentPlayer_block == self.currentPlayer_action:
                ## we have returned to the acting player, indicating that no one blocked the action
                self.DEBUG("action was not challenged by any player")
                self.performAttemptedAction()
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                self.changeState("Action")
                ret_observation = self.getObservation(self.currentPlayer_action)
            
            else:
                if action[2] == 1: 
                    self.DEBUG("player", self.playerList[self.currentPlayer_block].Name, "is attempting to block current action,", self.attemptedAction)
                    self.changeState("Challenge_block")
                    ret_observation = self.getObservation(self.currentPlayer_action)
                elif action[2] == 0:
                    ## we dont change state, just move to the next player and let them block if they want
                    self.currentPlayer_block = (self.currentPlayer_block + 1) % self.nPlayers
                    ret_observation = self.getObservation(self.currentPlayer_block)
                
        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase 
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        ##### TARGETTED BLOCKING STATE #####
        elif(self.gameState == "Blocking_target"): ## target of an action can attempt to block it 
            if action[2] == 1: 
                self.DEBUG("player", self.playerList[self.currentPlayer_block].Name, "is attempting to block current action,", self.attemptedAction)
                self.changeState("Challenge_block")
                ret_observation = self.getObservation(self.currentPlayer_action)
            else:
                self.DEBUG("action was not challenged by",self.playerList[self.currentPlayer_block].Name)
                self.performAttemptedAction()
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                self.changeState("Action")
                ret_observation = self.getObservation(self.currentPlayer_action)
        
        ##### GENERAL CHALLENGE STATE #####
        elif(self.gameState == "Challenge_general"): ## any player can challenge the attempted action
            if self.currentPlayer_block != self.currentPlayer_action:
                ## have returned back to acting player indicating no one blocked the action
                self.DEBUG("action was not challenged by any player")
                self.performAttemptedAction()
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                self.changeState("Action")
                ret_observation = self.getObservation(self.currentPlayer_action)

            else:
                if action[3] == 1: 
                    self.DEBUG("player", self.playerList[self.currentPlayer_challenge].Name, "is challenging", self.playerList[self.currentPlayer_action].Name, "on their action,", self.attemptedAction)
                    if self.challenge(self.currentPlayer_challenge, self.currentPlayer_action, actions[self.attemptedAction]["needs"]) == "failed":
                        self.performAttemptedAction()
                        self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                        self.changeState("Action")
                        ret_observation = self.getObservation(self.currentPlayer_action)
                elif action[3] == 0:
                        ## we dont change state, just move to the next player and let them block if they want
                        self.currentPlayer_challenge = (self.currentPlayer_challenge + 1) % self.nPlayers
                        ret_observation = self.getObservation(self.currentPlayer_challenge)
                        
        
        ##### TARGETTED CHALLENGE STATE #####
        elif(self.gameState == "Challenge_target"): ## target of an action can challenge it 
            if action[3] == 1: 
                self.DEBUG("player", self.playerList[self.currentPlayer_challenge].Name, "is attempting to challenge current action,", self.attemptedAction)
                if self.challenge(self.currentPlayer_challenge, self.currentPlayer_action, actions[self.attemptedAction]["needs"]) == "failed":
                    self.performAttemptedAction()
            else:
                self.DEBUG("action was not challenged by",self.playerList[self.currentPlayer_challenge].Name)
                self.performAttemptedAction()
                
            self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
            self.changeState("Action")
            ret_observation = self.getObservation(self.currentPlayer_action)
        
        ##### CHALLENGE BLOCKING STATE #####
        elif(self.gameState == "Challenge_block"): ## initial action taking player can challenge an attempt to block their action
            if action[4] == 1: 
                self.DEBUG("player", self.playerList[self.currentPlayer_action].Name, "is challenging the attempt by",self.playerList[self.currentPlayer_block].Name, "to block their action,", self.attemptedAction)
                self.challenge(self.currentPlayer_action, self.currentPlayer_block, *actions[self.attemptedAction]["blockedBy"])
            else:
                self.DEBUG("player", self.playerList[self.currentPlayer_action].Name, "accepts attempt by",self.playerList[self.currentPlayer_block].Name, "to block their action,", self.attemptedAction)
        
            self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
            self.changeState("Action")
            ret_observation = self.getObservation(self.currentPlayer_action)

        ##### BLOCK OR CHALLENGE STATE #####
        elif(self.gameState == "Block_or_Challenge"): ## target of an action can either block or challenge the action
            return
        else:
            self.ERROR("Something has gone wrong, have ended up in an undefined state:", self.gameState)
            raise Exception()
        
        return (ret_observation, ret_reward, ret_terminated, ret_truncated, ret_info)

gym.register("Coup_Game", Game)
game = gym.make("Coup_Game", nPlayers=4)
game.reset()

for _ in range(10):
    game.step(game.action_space.sample()) ##game.action_space.sample())


