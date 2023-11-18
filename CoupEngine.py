from Logging import *
import gym
import numpy as np
from Cards import *
from Player import *

## number of cards dealt to each player
MAX_CARDS = 2

## possible actions players can take
##         Action        | Card Needed          | Blocked by                           | Cost      | isTargeted
actions = {"Income":      {"needs":"",           "blockedBy":[""],                      "cost":0,   "targeted":False},
           "ForeignAid": {"needs":"",           "blockedBy":["Duke"],                  "cost":0,   "targeted":False},
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
        ### TODO: I think 3rd and 4th variable should be merged into one with 3 options: none, block, and challenge 
        ###  so that situations where either block or challenge are valid can be dealt with more easily
        ###  in situations where only one is valid, can easily just mask out the other one. 
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
        ## TODO: Add an observation for which player is targetting this player with an action
        ## TODO: Add an observation for which action is currently being attempted and by who

        ## initialise the players
        self.TRACE("  Creating players")
        self.playerList = []
        self.nPlayers = nPlayers
        for _ in range(nPlayers):
            self.playerList.append(Player(nCards = MAX_CARDS, logger = self.logger))

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
        self.activePlayer = 0
        self.action_target = 999
        self.currentPlayer_block = 999
        self.currentPlayer_challenge = 999

        ret_info = {}

        ret_info["mask"] = self.getMask(self.activePlayer)
        ret_info["activePlayer"] = self.activePlayer
        return (self.getObservation(self.activePlayer), 0, False, False, ret_info)

    ## Wrappers for logger functions for the logger specific to this game
    def ERROR(self, *messages): self.logger.error(*messages)
    def WARN(self, *messages): self.logger.warn(*messages)
    def INFO(self, *messages): self.logger.info(*messages)
    def DEBUG(self, *messages): self.logger.debug(*messages)
    def TRACE(self, *messages): self.logger.trace(*messages)

    ############################################################################################
    #### These are the functions that actually perform the actions specified by the players ####
    ############################################################################################
    def _Income(self, p):
        player = self.playerList[p]
        self.INFO(" Player: ", player.Name, "Action: Income")
        player.giveCoins(1)

        player.giveReward(1)

    def _ForeignAid(self, p):
        player = self.playerList[p]
        self.INFO(" Player: ", player.Name, "Action: ForeignAid")
        player.giveCoins(2)
        
        player.giveReward(2)

    def _Coup(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.INFO(" Player: ", player1.Name, "Action: Income, Target: ", player2.Name)
        player1.takeCoins(actions["Coup"]["cost"])
        player2.loseInfluence()
        
        player1.giveReward(10)

    def _Tax(self, p):
        player = self.playerList[p]
        self.INFO(" Player: ", player.Name, "Action: Tax")
        player.giveCoins(3)
        player.giveReward(3)

    def _Steal(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.INFO(" Player: ", player1.Name, "Action: Steal, Target: ", player2.Name)

        ## do this to avoid trying to steal 2 coins when target player doesn't have enoug
        ## this should probably be masked out but I'm not sure how to do that easily as it requires masking a specific combination of actions, (steal and specific players)
        ## BUT it's probably ok since the agents should learn to e.g. not steal from someone with 0 coins
        coinsToSteal = min(player2.Coins, 2)

        player2.takeCoins(coinsToSteal)
        player1.giveCoins(coinsToSteal)
        
        player2.giveReward(-coinsToSteal)
        player1.giveReward(coinsToSteal)

    def _Assasinate(self, p1, p2):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.INFO(" Player: ", player1.Name, "Action: Assassinate, Target: ", player2.Name)
        player1.takeCoins(3)
        player2.loseInfluence()

        player1.giveReward(10)

    def _Exchange():
        raise Exception("ERROR: Exchange() not implemented yet")
        ## not yet implemented

    def _Examine():
        raise Exception("ERROR: Exchange() not implemented yet")
        ## not yet implemented
    
    def performAttemptedAction(self):
        fn = self.__getattribute__("_" + self.attemptedAction) ## get the function corresponding to the attempted action

        if actions[self.attemptedAction]["targeted"]:
            fn(self.currentPlayer_action, self.action_target)
        else:
            fn(self.currentPlayer_action)
    


    
    def checkStatus(self):
        ## check the status of all the of all players
        ## for each one, check if all their cards are dead, if so, kill that player
        ## if all but one player is dead, they win and we're done
        ## returns true if game is finished, false otherwise

        if self.gameState == "Rewards":
            ## if we've already checked and are finished we can just return here
            return True 
        
        ## first check each player
        for player in self.playerList:
            allCardsDead = True
            for cardId in range(MAX_CARDS):
                if player.CardStates[cardId] == "Alive":
                    allCardsDead = False

            if(allCardsDead & player.isAlive):
                player.kill()

        aliveCount = 0
        for player in self.playerList:
            if player.isAlive:
                aliveCount += 1

        if aliveCount == 1:
            self.INFO("=========== GAME OVER ===========")
            
            for player in self.playerList:
                if player.isAlive:
                    self.INFO("")
                    self.INFO("  ** Player", player, "Wins! **")
                    player.giveReward(50)

                    ## move to special reward round state
                    self.gameState = "Rewards"
                    self.currentPlayer_Reward = 0
                    self.activePlayer = 0
                    return True
            
        return False

                
    
    def getMask(self, playerIdx):
        ## get action space mask for player at index playerIdx in this games player list
        self.DEBUG("Getting action mask for player", self.playerList[playerIdx].Name, "at index", playerIdx)

        ## make the action space spec
        ## 1st variable is which action to take in the "action" phase of the game
        ## 2nd variable is which other player to target if applicable
        ## 3rd variable is whether or not to attempt to block current attemted action in the "blocking" phase 
        ## 4th variable is whether or not to challenge the acting player in the "challenge" phase
        ## 5th variable is whether or not to challenge the attempted block
        maskList = []
        maskList.append(np.ndarray((len(actions.keys())), dtype=np.int8))
        maskList.append(np.ndarray((self.nPlayers-1,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))
        maskList.append(np.ndarray((2,), dtype=np.int8))

        if(self.gameState == "Action"):
            if self.playerList[playerIdx].Coins < 10:
                ## if player has >= 10 they can only perform a coup
                ## and their only choice is which player to target
                ## so we leave all mask values for actions at their default value of 0
                for i, action in enumerate(actions.keys()):
                    if self.playerList[playerIdx].Coins >= actions[action]["cost"]:
                        maskList[0][i] = 1 
                    else: 
                        maskList[0][i] = 0

                    ## currently unimplemented actions
                    if action in ["Exchange", "Examine"]:
                        maskList[0][i] = 0

            ## can only target players that are alive
            for id in range(1, self.nPlayers):
                if self.playerList[(id + playerIdx) % self.nPlayers].isAlive:
                    maskList[1][id - 1] = 1
                else:
                    maskList[1][id - 1] = 0


        else:
            maskList[0][:] = 0
            maskList[1][:] = 0
        
        if(self.gameState in ["Blocking_general", "Blocking_target"]):
            maskList[2][:] = 1
        else:
            maskList[2][:] = 0

        if(self.gameState in ["Challenge_general", "Challenge_target"]):
            maskList[3][:] = 1
        else:
            maskList[3][:] = 0

        if(self.gameState == "Challenge_block"):
            maskList[4][:] = 1
        else:
            maskList[4][:] = 0

        ## If in final reward state, player cant do anything
        if(self.gameState == "Rewards"):
            for i in range(len(maskList)):
                maskList[i][:] = 0

        ## If dead, player cant do anything
        if(not self.playerList[playerIdx].isAlive):
            for i in range(len(maskList)):
                maskList[i][:] = 0

        self.DEBUG("Mask: ", maskList)
        return tuple(maskList)
    
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

    def swapCard(self, p, card):
        ## take card from player, return to deck, shuffle then draw a new one
        player=self.playerList[p]
        player.takeCard(card)
        self.INFO("Player", player.Name, "Swapping card", card)
        self.Deck.returnCard(card)
        self.Deck.shuffle()
        newCard = self.Deck.draw()
        player.giveCard(newCard)
        self.INFO("       was swapped for", newCard)

    def challenge(self, p1, p2, *cards):
        player1, player2 = self.playerList[p1], self.playerList[p2]
        self.DEBUG("Player", player1.Name, "challenging Player", player2.Name, "on having one of", *cards)

        ## first shuffle order of the cards just to be extra fair
        cardList = [*cards]
        np.random.shuffle(cardList)
        for card in cardList:
            if player2.checkCard(card):
                self.INFO("Challenge failed,", player2.Name, "had a", card)
                ## according to rules, player needs to return the card and get a new one
                self.swapCard(p2, card)
                player1.loseInfluence()
                return "failed"

        ## if made it to this point, player2 didnt have any of the specified cards
        self.INFO("Challenge succeeded,", player2.Name, "did not have any of", *cards)
        player2.loseInfluence()

        return "succeeded"

    def step(self, action):
        self.INFO("")
        self.INFO("##### Stepping #####")
        self.INFO("gameState:",self.gameState)
        self.DEBUG("specified actions:", action)

        self.DEBUG("Active player", self.playerList[self.activePlayer])

        ## tings to return at the end of the step
        ret_observation = None
        ret_reward = 0
        ret_terminated = False
        ret_truncated = False
        ret_info = {}

        ## set this so that on the outside, we know which player we should give the reward to 
        ret_info["activePlayer"] = self.activePlayer
        ret_reward = self.playerList[self.activePlayer].claimReward()

        ## check what state we are in 

        ##### ACTION STATE #####
        if(self.gameState == "Action"):
            if not self.playerList[self.currentPlayer_action].isAlive:
                self.INFO(self.playerList[self.currentPlayer_action].Name, "is dead! skipping their action")
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                self.activePlayer = self.currentPlayer_action

            else:
                self.attemptedAction = actionNames[action[0]]
                self.INFO("Player", self.playerList[self.currentPlayer_action].Name, "is attempting action", self.attemptedAction)
                
                blockable = False
                targetted = False
                challengable = False
                ## first check if this action has a target
                if actions[self.attemptedAction]["targeted"]:
                    self.action_target = (self.currentPlayer_action + 1 + action[1]) % self.nPlayers
                    self.INFO("Targetting player", self.playerList[self.action_target].Name, "at index", self.action_target)
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
                    self.activePlayer = self.currentPlayer_block

                elif challengable:
                    if targetted: 
                        self.currentPlayer_challenge = self.action_target
                        self.changeState("Challenge_target")
                    else:
                        self.currentPlayer_challenge = (self.currentPlayer_action + 1) % self.nPlayers
                        self.changeState("Challenge_general")
                    self.activePlayer = self.currentPlayer_challenge
            
                else:
                    self.performAttemptedAction()
                    self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                    self.activePlayer = self.currentPlayer_action

        ##### GENERAL BLOCKING STATE #####
        elif(self.gameState == "Blocking_general"): ## state in which any player can attempt to block the attempted action
            if not self.playerList[self.currentPlayer_block].isAlive:
                self.INFO(self.playerList[self.currentPlayer_block].Name, "is dead so they can't really block anything")
                self.currentPlayer_block = (self.currentPlayer_block + 1) % self.nPlayers
                self.activePlayer = self.currentPlayer_block

            else:
                if self.currentPlayer_block == self.currentPlayer_action:
                    ## we have returned to the acting player, indicating that no one blocked the action
                    self.INFO("action was not challenged by any player")
                    self.performAttemptedAction()
                    self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                    self.changeState("Action")
                    self.activePlayer = self.currentPlayer_action
                
                else:
                    if action[2] == 1: 
                        self.INFO("player", self.playerList[self.currentPlayer_block].Name, "is attempting to block current action,", self.attemptedAction)
                        self.changeState("Challenge_block")
                        self.activePlayer = self.currentPlayer_action
                    elif action[2] == 0:
                        ## we dont change state, just move to the next player and let them block if they want
                        self.INFO("player", self.playerList[self.currentPlayer_block].Name, "did not try to block current action,", self.attemptedAction)
                        self.currentPlayer_block = (self.currentPlayer_block + 1) % self.nPlayers
                        self.activePlayer = self.currentPlayer_block
                
        ##### TARGETTED BLOCKING STATE #####
        elif(self.gameState == "Blocking_target"): ## target of an action can attempt to block it 
            if action[2] == 1: 
                self.INFO("player", self.playerList[self.currentPlayer_block].Name, "is attempting to block current action,", self.attemptedAction)
                self.changeState("Challenge_block")
                self.activePlayer = self.currentPlayer_action
            else:
                self.INFO("action was not challenged by",self.playerList[self.currentPlayer_block].Name)
                self.performAttemptedAction()
                self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                self.changeState("Action")
                self.activePlayer = self.currentPlayer_action
        
        ##### GENERAL CHALLENGE STATE #####
        elif(self.gameState == "Challenge_general"): ## any player can challenge the attempted action
            if not self.playerList[self.currentPlayer_challenge].isAlive:
                self.INFO(self.playerList[self.currentPlayer_challenge].Name, "is dead so they can't really challenge anything")
                self.currentPlayer_challenge = (self.currentPlayer_challenge + 1) % self.nPlayers
                self.activePlayer = self.currentPlayer_challenge
               
            else:
                if self.currentPlayer_challenge == self.currentPlayer_action:
                    ## have returned back to acting player indicating no one blocked the action
                    self.INFO("action was not challenged by any player")
                    self.performAttemptedAction()
                    self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                    self.changeState("Action")
                    self.activePlayer = self.currentPlayer_action

                else:
                    if action[3] == 1: 
                        self.INFO("player", self.playerList[self.currentPlayer_challenge].Name, "is challenging", self.playerList[self.currentPlayer_action].Name, "on their action,", self.attemptedAction)
                        if self.challenge(self.currentPlayer_challenge, self.currentPlayer_action, actions[self.attemptedAction]["needs"]) == "failed":
                            self.performAttemptedAction()
                        
                        ## If a challenge happened, it will be resolved straight away so we move back to the action state
                        self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
                        self.changeState("Action")
                        self.activePlayer = self.currentPlayer_action

                    elif action[3] == 0:
                            self.INFO("player", self.playerList[self.currentPlayer_challenge].Name, "did not attempt to challenge")
                            ## we dont change state, just move to the next player and let them block if they want
                            self.currentPlayer_challenge = (self.currentPlayer_challenge + 1) % self.nPlayers
                            self.activePlayer = self.currentPlayer_challenge
                            
        
        ##### TARGETTED CHALLENGE STATE #####
        elif(self.gameState == "Challenge_target"): ## target of an action can challenge it 
            if action[3] == 1: 
                self.INFO("player", self.playerList[self.currentPlayer_challenge].Name, "is attempting to challenge current action,", self.attemptedAction)
                if self.challenge(self.currentPlayer_challenge, self.currentPlayer_action, actions[self.attemptedAction]["needs"]) == "failed":
                    self.performAttemptedAction()
            else:
                self.INFO("action was not challenged by",self.playerList[self.currentPlayer_challenge].Name)
                self.performAttemptedAction()
                
            self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
            self.changeState("Action")
            self.activePlayer = self.currentPlayer_action
        
        ##### CHALLENGE BLOCKING STATE #####
        elif(self.gameState == "Challenge_block"): ## initial action taking player can challenge an attempt to block their action
            if action[4] == 1: 
                self.INFO("player", self.playerList[self.currentPlayer_action].Name, "is challenging the attempt by",self.playerList[self.currentPlayer_block].Name, "to block their action,", self.attemptedAction)
                self.challenge(self.currentPlayer_action, self.currentPlayer_block, *actions[self.attemptedAction]["blockedBy"])
            else:
                self.INFO("player", self.playerList[self.currentPlayer_action].Name, "accepts attempt by",self.playerList[self.currentPlayer_block].Name, "to block their action,", self.attemptedAction)
        
            self.currentPlayer_action = (self.currentPlayer_action + 1) % self.nPlayers
            self.changeState("Action")
            self.activePlayer = self.currentPlayer_action

        ##### BLOCK OR CHALLENGE STATE #####
        elif(self.gameState == "Block_or_Challenge"): ## target of an action can either block or challenge the action
            return
        
        ##### REWARDS STATE #####
        elif(self.gameState == "Rewards"): ## target of an action can either block or challenge the action
            player = self.playerList[self.activePlayer]
            reward = player.claimReward()
            self.INFO("Player", player.Name, "Final reward:", reward)
            
            self.currentPlayer_Reward = (self.currentPlayer_Reward + 1) % self.nPlayers
            self.activePlayer = self.currentPlayer_Reward
        
            ret_reward = reward

            if self.currentPlayer_Reward == 0:
                self.INFO("========= DONE HANDING OUT REWARDS ==========")
                ret_terminated = True ## And we're DONE!

        else:
            self.ERROR("Something has gone wrong, have ended up in an undefined state:", self.gameState)
            raise Exception()
        
        
        if not ret_terminated:
            self.checkStatus()
        
        ret_info["mask"] = self.getMask(self.activePlayer)
        
        return (self.getObservation(self.activePlayer), ret_reward, ret_terminated, ret_truncated, ret_info)
