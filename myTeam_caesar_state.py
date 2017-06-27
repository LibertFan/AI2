# -*- coding: utf-8 -*-
# __author__ = 'siyuan'

# myTeam_caesar.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions, Actions
from util import nearestPoint
from decimal import Decimal
from itertools import product
import copy

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
  
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    if firstIndex < 2:
        first = 'Caesar'
    elif firstIndex < 4:
        first = 'Caesar1'
    if secondIndex < 2:
        second = 'Caesar'
    else:
        second = 'Caesar1'
    print first
    print second
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
class QJT( CaptureAgent ):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.allies = self.getTeam( gameState)
        self.enemies = self.getOpponents( gameState )
        self.RedBorder = 
        self.BlueBorder = 

    def chooseAction( self, gameState ):

        MaxMinScore = -9999
        ChosedAlliesAction = None
        for AlliesAction in  list( product( list( product( self.allies[0], gameState.getLegalActions( self.allies[0] ) ) ),\
                                             list( product( self.allies[1], gameState.getLegalActions( self.allies[1] ) ) ) ) ):
            MinScore = 9999    
            ChosedEnemiesAction = None
            for EnemiesAction in  list( product( list( product( self.enemies[0], gameState.getLegalActions( self.enemies[0] ) ) ),\
                                                 list( product( self.enemies[1], gameState.getLegalActions( self.enemies[1] ) ) ) ) ):

                IndexActions = AlliesAction + EnemiesAction
                CurrentScore = self.evaluate( gameState, IndexActions )

                if CurrentScore < MinScore:
                    MinScore = CurrentScore
                    ChosedEnemiesAction = EnemiesAction     
            
            if MinScore!= 9999 and MinScore > MaxMinScore:
                MaxMinScore = MinScore
                ChosedAlliesAction = AlliesAction

        index = self.allies.index( self.index )
        BestAction = ChosedAlliesAction[ index ][1]


        """
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        """

        return BestAction

    def getSuccessor(self, gameState, actions):
        """
        Finds the next successor which is a grid position (location tuple).
        actions are respectively for index: 0, 1, 2, 3 
        !!!!!!! Attention !!!!!!!
        GameOver!
        """    
        CurrentGameState = copy.deepcopy( gameState )
        agentMoveInfo = []
        for agentIndex, action in actions:
            isAgentPacman = CurrentGameState.getAgentState( agentIndex ).isPacman
            agentMoveInfo.append( ( isAgentPacman, agentIndex, action ) ) 
        agentMoveOrder = sorted( agentMoveInfo, key=lambda x:( x[0], x[1] ) ) 

        deadAgentList = []
        for index, ( isAgentPacman, agentIndex, action ) in enumerate( agentMoveOrder ):
            if agentIndex in deadAgentList:
                CurrentGameState, deadAgents = CurrentGameState.generateSuccessor( agentIndex, "Stop", True )
                deadAgentList.extend( deadAgents ) 
            else:    
                CurrentGameState, deadAgents = CurrentGameState.generateSuccessor( agentIndex, action, True )
                deadAgentList.extend( deadAgents )

        return CurrentGameState 

    def evaluate(self, gameState, actions):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, actions)
        weights = self.getWeights()
        return features * weights

    def isIntercept( self, gameState, escapeAgentIndexList, chaseAgentIndexList, ObjsType = None ):
        ### First, we need to judge that whether the escapeAgentIndexList
        ### If
        isRed = gameState.isOnRedTeam( chaseAgentIndexList[0] )
        if ObjsType == 0:
        ### escapeAgents try to escape back to their own territoty!
        ### is chaseAgent able to stop them ?
            if isRed:
                Objs = self.RedBorder
            else:
                Objs = self.BlueBorder
        elif ObjsType == 1:
        ### escapeAgents try to eat the capsules!
        ### is chaseAgent able to stop them ?
            if isRed:
                Objs = gameState.getBlueCapsules()
            else:
                Objs = gameState.getRedCapsules()
        else:
            return None

        escapeAgentPositionList = [ gameState.getAgentState( agentIndex ).getPositions() for agentIndex in escapeAgentIndexList ]
        chaseAgentPositionList = [ gameState.getAgetState( agentIndex ).getPositions() for agentIndex in chaseAgentIndexList ]
        
        escapeAgentToBorderDistanceList = []
        for escapeAgentPos in escapeAgentPositionList:
            escapeAgentToBorderDistance = []
            for pos in Objs:
                escapeAgentToBorderDistance.append( self.getMazeDistance( pos, escapeAgentPos ) )
                escapeAgentToBorderDistanceList.append( escapeAgentToBorderDistance )

        chaseAgentToBorderDistanceList = []    
        for chaseAgentPos in chaseAgentPositionList:                    
            chaseAgentToBorderDistance = []
            for pos in Objs:
                chaseAgentToBorderDistance.append( self.getMazeDistance( pos, chaseAgentPos ) )
            chaseAgentToBorderDistanceList.append( chaseAgentToBorderDistance )

        interceptDistanceList = []
        for escapeAgentToBorderDistance in escapeAgentToBorderDistanceList:
            interceptDistance = []
            for PosIndex in range( len( Objs ) ):
                escapeAgentDistance = escapeAgentToBorderDistance[ PosIndex ]
                currentInterceptDistanceList = [ escapeAgentDistance - chaseAgentToBorderDistance[ PosIndex ]
                                          for chaseAgentToBorderDistance in chaseAgentToBorderDistanceList ]
                interceptDistance.append( max( currentInterceptDistanceList.append( -1 ) ) )    
            interceptDistanceList.append( min( interceptDistance ) )     
        
        isInterceptList = [ distance >= 0 for distance in interceptDistanceList ]

        return zip( escapeAgentIndexList, interceptDistanceList, isInterceptList ) 

    def getActionFeatures( self, gameState, actions):
    
    def getStateFeatures( self, gameState, actions):

    def getPacmanFeatures( self, gameState ):

    def getGhostFeatures( self, gameState, agentIndex ):
        features = util.Counter()
        agentState = gameState.getAgentState( agentIndex )
        isScaring = agentState.scaredTimer > 0
        if isScaring:
            Invaders = [ agentIndex if gameState.getAgentState( agentIndex ).isPacman for agentIndex in self.enemies ]
            minDistancerToInvaders = []
            
            for pos in self.



        else:





    def getFeatures(self, gameState, actions):
        """
        Returns a counter of features for the state
        """
        for agentIndex, action in actions:
            if agentIndex == self.index:                
                AgentAction = action
                break

        features = util.Counter()
        ### score for action
        if action == Directions.STOP:
            features["stopped"] = 1.0
        else:
            features["stopped"] = 0.0
        
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if rev == agentAction:
            features["reverse"] = 1.0
        else:
            features["reverse"] =  0.0

        OldAgentState = gameState.getAgentState( self.index ) 
        NewState = self.getSuccessor( gameState, actions )          
        NewAgentState = NewState.getAgentState( self.index )

        # num_eat_foods <= 1
        for agentIndex in self.allies:
            OldAgentState = gameState.getAgentState( agentIndex )
            NewAgentState = NewState.getAgentState( agentIndex )

        for agentIndex in self.enemies:    
            OldAgentState = gameState.getAgentState( agentIndex )
            NewAgentState = NewState.getAgentState( agentIndex )
           

        num_eat_foods = NewAgentState.numCarrying + NewAgentState.numReturned - OldAgentState.numCarrying - OldAgentState.numReturned
        is_eat_capsule = NewAgentState.numCapsules - OldAgentState.numCapsules
        ### eat-foods may less than zero which means that it was wipe out by enemies
        features["carry-new-foods"] = NewAgentState.numCarrying - OldAgentState.numCarrying 
        features["return-new-foods"] = NewAgentState.numReturn - OldAgentState.numReturn
        #features["eat-foods"] = num_eat_foods
        features["eat-capsules"] = is_eat_capsule
       
	### score for state

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}
     
     


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        print "self.index", self.index
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """ 
        LegalActions = []       
        for i in range(4):
			LegalActions.append( gameState.getLegalActions( i ) )     

        NewLegalActions = [] 
        for a0 in LegalActions[0]:
	    for a1 in LegalActions[1]:
		for a2 in LegalActions[2]:
		    for a3 in LegalActions[3]:
  			NewLegalActions.append( ( a0, a1, a2, a3 ) )


        
        min() 
        values = [ self.evaluate(gameState, actions ) for actions in NewLegalActions ]
        ### Use min-max to choose the best action!
        maxValue = max(values)
    	bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
                bestDist = 9999
                for action in actions:
                        successor = self.getSuccessor(gameState, action)
                        pos2 = successor.getAgentPosition(self.index)
                        dist = self.getMazeDistance(self.start,pos2)
                        if dist < bestDist:
                                bestAction = action
                                bestDist = dist
                return bestAction

        return random.choice(bestActions) 

    def getSuccessor(self, gameState, actions):
        """
        Finds the next successor which is a grid position (location tuple).
        actions are respectively for index: 0, 1, 2, 3 
        """       
        agentMoveInfo = []
        for agentIndex, action in enumerate( actions ):
            isAgentPacman = gameState.getAgentState( agentIndex ).isPacman
            agentMoveInfo.append( ( isAgentPacman, agentIndex, action ) ) 
        agentMoveOrder = sorted( agentMoveInfo, key=lambda x:( x[0], x[1] ) ) 

        deadAgentList = []
        for index, ( isAgentPacman, agentIndex, action ) in enumerate( agentMoveOrder ):
            if agentIndex in deadAgentList:
                CurrentGameState, deadAgents = gameState.generateSuccessor( agentIndex, "Stop", True )
                deadAgentList.extend( deadAgents ) 
            else:    
                CurrentGameState, deadAgents = gameState.generateSuccessor( agentIndex, action, True )
                deadAgentList.extend( deadAgents )

        return gameState				

    def evaluate(self, gameState, actions):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, actions)
        weights = self.getWeights(gameState, actions)
        return features * weights

    def getFeatures(self, gameState, actions):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, actions)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class Caesar(ReflexCaptureAgent):
    def getFeatures(self, state, action):
        food = self.getFood(state)
        foodList = food.asList()
        walls = state.getWalls()
        isPacman = self.getSuccessor(state, action).getAgentState(self.index).isPacman

        # Zone of the board agent is primarily responsible for
        zone = (self.index - self.index % 2) / 2

        teammates = [state.getAgentState(i).getPosition() for i in self.getTeam(state)]
        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]
        chasers = [a for a in opponents if not (a.isPacman) and a.getPosition() != None]
        prey = [a for a in opponents if a.isPacman and a.getPosition() != None]

        features = util.Counter()
        if action == Directions.STOP:
            features["stopped"] = 1.0
        # compute the location of pacman after he takes the action
        x, y = state.getAgentState(self.index).getPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        for g in chasers:
            if (next_x, next_y) == g.getPosition():
                if g.scaredTimer > 0:
                    features["eats-ghost"] += 1
                    features["eats-food"] += 2
                else:
                    features["#-of-dangerous-ghosts-1-step-away"] = 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
            elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                if g.scaredTimer > 0:
                    features["#-of-harmless-ghosts-1-step-away"] += 1
                elif isPacman:
                    features["#-of-dangerous-ghosts-1-step-away"] += 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
        if state.getAgentState(self.index).scaredTimer == 0:
            for g in prey:
                if (next_x, next_y) == g.getPosition:
                    features["eats-invader"] = 1
                elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                    features["invaders-1-step-away"] += 1
        else:
            for g in opponents:
                if g.getPosition() != None:
                    if (next_x, next_y) == g.getPosition:
                        features["eats-invader"] = -10
                    elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                        features["invaders-1-step-away"] += -10

        for capsule_x, capsule_y in state.getCapsules():
            if next_x == capsule_x and next_y == capsule_y and isPacman:
                features["eats-capsules"] = 1.0
        if not features["#-of-dangerous-ghosts-1-step-away"]:
            if food[next_x][next_y]:
                features["eats-food"] = 1.0
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                myFood = []
                for food in foodList:
                    food_x, food_y = food
                    if (food_y > zone * walls.height / 3 and food_y < (zone + 1) * walls.height / 3):
                        myFood.append(food)
                if len(myFood) == 0:
                    myFood = foodList
                myMinDist = min([self.getMazeDistance((next_x, next_y), food) for food in myFood])
                if myMinDist is not None:
                    features["closest-food"] = float(myMinDist) / (walls.width * walls.height)

        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {'eats-invader': 5, 'invaders-1-step-away': 0, 'teammateDist': 1.5, 'closest-food': -1,
                'eats-capsules': 10.0, '#-of-dangerous-ghosts-1-step-away': -20, 'eats-ghost': 1.0,
                '#-of-harmless-ghosts-1-step-away': 0.1, 'stopped': -5, 'eats-food': 1}

class QJT( ReflexCaptureAgent ):
    def getFeatures( self, state, actions ):   
        features = util.Counter()
        agentAction = actions[ self.index ]
        ### score for action
        if action == Directions.STOP:
            features["stopped"] = 1.0
        else:
            features["stopped"] = 0.0
        
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if rev == agentAction:
            features["reverse"] = 1.0
        else:
            features["reverse"] =  0.0

        OriginState = copy.deepcopy( state ) 
        OldAgentState = OriginState.getAgentState( self.index )
        agentMoveInfo = []
        for agentIndex, action in enumerate( actions ):
            isAgentPacman = OriginState.getAgentState( agentIndex ).isPacman
            agentMoveInfo.append( ( isAgentPacman, agentIndex, action ) ) 
        agentMoveOrder = sorted( agentMoveInfo, key = lambda x:( x[0], x[1] ) ) 

        deadAgentList = []
        for index, ( isAgentPacman, agentIndex, action ) in enumerate( agentMoveOrder ):
            if agentIndex in deadAgentList:
                OriginState, deadAgents = OriginState.generateSuccessor( agentIndex, "Stop", True )
                deadAgentList.extend( deadAgents ) 
            else:    
                OriginState, deadAgents = OriginState.generateSuccessor( agentIndex, action, True )
                deadAgentList.extend( deadAgents )
        NewAgentIndex = OriginState.getAgentState( self.index )

        # num_eat_foods <= 1
        num_eat_foods = NewAgentState.numCarrying + NewAgentState.numReturned - OldAgentState.numCarrying - OldAgentState.numReturned
        is_eat_capsule = NewAgentState.numCapsules - OldAgentState.numCapsules
        
        features["eat-foods"] = num_eat_foods
        features["eat-capsules"] = is_eat_capsule
       
         


	### score for state

    def getWeights( self, gameState, action ):
	return {}


class Caesar1(ReflexCaptureAgent):
    def getFeatures(self, state, action):
        food = self.getFood(state)
        foodList = food.asList()
        walls = state.getWalls()
        isPacman = self.getSuccessor(state, action).getAgentState(self.index).isPacman

        # Zone of the board agent is primarily responsible for
        zone = (self.index - self.index % 2) / 2

        teammates = [state.getAgentState(i).getPosition() for i in self.getTeam(state)]
        opponents = [state.getAgentState(i) for i in self.getOpponents(state)]
        chasers = [a for a in opponents if not (a.isPacman) and a.getPosition() != None]
        prey = [a for a in opponents if a.isPacman and a.getPosition() != None]

        features = util.Counter()
        if action == Directions.STOP:
            features["stopped"] = 1.0
        # compute the location of pacman after he takes the action
        x, y = state.getAgentState(self.index).getPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        for g in chasers:
            if (next_x, next_y) == g.getPosition():
                if g.scaredTimer > 0:
                    features["eats-ghost"] += 1
                    features["eats-food"] += 2
                else:
                    features["#-of-dangerous-ghosts-1-step-away"] = 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
            elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                if g.scaredTimer > 0:
                    features["#-of-harmless-ghosts-1-step-away"] += 1
                elif isPacman:
                    features["#-of-dangerous-ghosts-1-step-away"] += 1
                    features["#-of-harmless-ghosts-1-step-away"] = 0
        if state.getAgentState(self.index).scaredTimer == 0:
            for g in prey:
                if (next_x, next_y) == g.getPosition:
                    features["eats-invader"] = 1
                elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                    features["invaders-1-step-away"] += 1
        else:
            for g in opponents:
                if g.getPosition() != None:
                    if (next_x, next_y) == g.getPosition:
                        features["eats-invader"] = -10
                    elif (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls):
                        features["invaders-1-step-away"] += -10

        for capsule_x, capsule_y in state.getCapsules():
            if next_x == capsule_x and next_y == capsule_y and isPacman:
                features["eats-capsules"] = 1.0
        if not features["#-of-dangerous-ghosts-1-step-away"]:
            if food[next_x][next_y]:
                features["eats-food"] = 1.0
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                myFood = []
                for food in foodList:
                    food_x, food_y = food
                    if (food_y > zone * walls.height / 3 and food_y < (zone + 1) * walls.height / 3):
                        myFood.append(food)
                if len(myFood) == 0:
                    myFood = foodList
                myMinDist = min([self.getMazeDistance((next_x, next_y), food) for food in myFood])
                if myMinDist is not None:
                    features["closest-food"] = float(myMinDist) / (walls.width * walls.height)

        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {'eats-invader': 5, 'invaders-1-step-away': 1, 'teammateDist': 1.5, 'closest-food': -1,
                'eats-capsules': 10.0, '#-of-dangerous-ghosts-1-step-away': -20, 'eats-ghost': 1.0,
                '#-of-harmless-ghosts-1-step-away': 0.1, 'stopped': -5, 'eats-food': 1}


