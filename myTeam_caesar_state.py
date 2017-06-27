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
import random, time, util, sys, copy
from game import Directions, Actions
from util import nearestPoint
from decimal import Decimal
from itertools import product

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent', Param_Weights_1 = None, Param_Weights_2 = None):
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
    return [eval("QJT")(firstIndex), eval("QJT")(secondIndex)]


##########
# Agents #
##########
global BestAction
BestAction = None

class QJT( CaptureAgent ):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.allies = self.getTeam( gameState)
        self.enemies = self.getOpponents( gameState )

        CurrentLayout = gameState.data.layout
        height = CurrentLayout.height
        width = gameState.data.layout.width
        #print "height:", height, "width:",width
        redline = width / 2 - 1
        blueline = redline 
        #print "redline:",redline,"blueline:",blueline
        self.RedBorder = []
        self.BlueBorder = []
        for i in range(1,height):
            Pos = ( redline, i )
            if not CurrentLayout.isWall( Pos):
                self.RedBorder.append( Pos)
            Pos = ( blueline, i )
            if not CurrentLayout.isWall( Pos):
                self.BlueBorder.append( Pos)

        self.red = gameState.isOnRedTeam( self.index )  
        if self.red:
            self.AllyBorder = self.RedBorder
            self.EnemyBorder = self.BlueBorder
        else:
            self.AllyBorder = self.BlueBorder
            self.EnemyBorder = self.RedBorder
        #print "RedBorder", self.RedBorder
        #print "BlueBorder", self.BlueBorder
        #print "AllyBorder", self.AllyBorder
        #print "EnemyBorder",self.EnemyBorder

    def chooseAction( self, gameState ):
        ### We need to make sure that the agent would not be eaten
        foodLeft = len(self.getFood( gameState ).asList())
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                ### the following getSuccessor need to change
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition( self.index )
                dist = self.getMazeDistance( self.start, pos2 )
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        global BestAction
        if BestAction is not None:
            bestAction = BestAction
            BestAction = None
            return bestAction

        MaxMinScore = -9999
        ChosedAlliesAction = None
        for AlliesAction in  list( product( list( product( [ self.allies[0], ], gameState.getLegalActions( self.allies[0] ) ) ),\
                                             list( product( [ self.allies[1], ], gameState.getLegalActions( self.allies[1] ) ) ) ) ):
            MinScore = 9999    
            ChosedEnemiesAction = None
            for EnemiesAction in  list( product( list( product( [ self.enemies[0], ], gameState.getLegalActions( self.allies[0] ) ) ),\
                                                 list( product( [ self.enemies[1], ], gameState.getLegalActions( self.enemies[1] ) ) ) ) ):

                IndexActions = AlliesAction + EnemiesAction
                print IndexActions
                CurrentScore = self.evaluate( gameState, IndexActions )

                if CurrentScore < MinScore:
                    MinScore = CurrentScore
                    ChosedEnemiesAction = EnemiesAction     
            
            if MinScore!= 9999 and MinScore > MaxMinScore:
                MaxMinScore = MinScore
                ChosedAlliesAction = AlliesAction

        index = self.allies.index( self.index )
        bestAction = ChosedAlliesAction[ index ][1]

        for ix in self.allies:
            if ix != self.index:
                break
        allyBestAction = ChosedAlliesAction[ix][1]
        global BesAction
        BestAction = allyBestAction

        return bestAction

    def getSuccessor(self, gameState, actions, returnDeadAgent = False):
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

        if not returnDeadAgent:
            return CurrentGameState 
        else:
            return CurrentGameState, deadAgentList

    def evaluate(self, gameState, actions):
        """
        Computes a linear combination of features and feature weights
        """
        Score = 0
        ActionFeatures = self.getActionFeatures(gameState, actions)
        succGameState = self.getSuccessor( gameState, actions)
        ActionScore = ActionFeatures * self.getWeights()
        Score += ActionScore
        for agentIndex in self.allies:
            AgentFeatures = self.getAgentFeatures( succGameState, agentIndex)
            AgentScore = AgentFeatures * self.getWeigts()
            Score += AgentScore
        return Score

    def isIntercept( self, gameState, escapeAgentIndexList, chaseAgentIndexList, ObjsType = None, scaredTimer = -9999):
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
        for escapeAgentPos in escapAgentPositionList:
            escapeAgentToBorderDistance = []
            for pos in Objs:
                escapeAgentToBorderDistance.append( self.getMazeDistance( pos, escapeAgentPos ) )
            escapeAgentToBorderDistanceeList.append( escapeAgentToBorderDistance )

        chaseAgentToBorderDistanceList = []    
        for chaseAgentPos in chaseAgentPositionList:                    
            chaseAgentToBorderDistance = []
            for pos in Objs:
                chaseAgentToBorderDistance.append( self.getMazeDistance( pos, AgentPos ) )
            chaseAgentToBorderDistanceList.append( chaseAgentToBorderDistance )

        interceptDistanceList = []
        for escapeAgentToBorderDistance in escapeAgentToBorderDistanceList:
            interceptDistance = []
            for PosIndex in range( len( Objs ) ):
                escapeAgentDistance = escapeAgentToBorderDistance[ PosIndex ]
                currentInterceptDistanceList = [ max( [ escapeAgentDistance - chaseAgentToBorderDistance[ PosIndex ], 2 * int( scaredTimer >= escapeDistance ) - 1 ] )
                                          for chaseAgentToBorderDistance in chaseAgentToBorderDistanceList ]
                interceptDistance.append( max( currentInterceptDistanceList.append( -1 ) ) )    
            interceptDistanceList.append( min( interceptDistance ) )     
        
        isInterceptList = [ distance >= 0 for distance in interceptDistanceList ]
        if scaredTimer <= 0:
            return zip( escapeAgentIndexList, interceptDistanceList, isInterceptList ) 
        else:
            return zip( escapeAgentIndexList, isInterceptList ) 

    def getActionFeatures( self, gameState, actions):
        for agentIndex, action in actions:
            if agentIndex == self.index:                
                agentAction = action
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

        #num_eat_foods = NewAgentState.numCarrying + NewAgentState.numReturned - OldAgentState.numCarrying - OldAgentState.numReturned
        ### eat-foods may less than zero which means that it was wipe out by enemies
        features["eat-new-food"] = int( NewAgentState.numCarrying - OldAgentState.numCarrying > 0 ) 
        features["return-new-food"] = int( NewAgentState.numReturned - OldAgentState.numReturned > 0 )
	features["return-new-food-number"] = NewAgentState.numReturned - OldAgentState.numReturned
        features["eat-capsules"] = int( NewAgentState.numCapsules - OldAgentState.numCapsules < 0 )

        ### CheckDeath
        _, deathAgents = self.getSuccessor( gameState, actions )
        index = 0
        for agentIndex in self.allies:
            if agentIndex in deadAgents:
                if gameState.getAgentState( agentIndex ).isPacman:
                    features[ "Ally-Pacman-Die" + str(index) ] = 1
                    features[ "Ally-Die-food" + str(index) ] = gameState.getAgentState( agentIndex ).numCarrying
                else:
                    features["Ally-Ghost-Die" + str(index)] = 1
                index += 1    
        
        index = 0
        for agentIndex in self.enemies:
            if agentIndex in deadAgents:
                if gameState.getAgentState( agentIndex ).isPacman:
                    features[ "Enemy-Pacman-Die" + str(index) ] = 1
                    features[ "Enemy-Pacman-Die-food" + str(index) ] = gameState.getAgentState( agentIndex ).numCarrying
                else:
                    features["Enemy-Ghost-Die" + str(index) ] = 1
                index += 1    

        return features

    def getAgentFeatures( self, gameState, agentIndex):
        return self.getAgentPacmnFeatures( gameState, agentIndex) + self.getGhostFeatures( gameState, agentIndex)

    def getAgentPacmanFeatures( self, gameState, agentIndex ):
        features = util.Counter()
        agentPosition = gameState.getAgentState( agentIndex ).getPosition()

        try:
            foodPositionList = self.getFood( gameState ).asList()
            miDistanceToFood = min([ self.getMazeDistance( food, agentPosition ) for foodPosition in foodPositionList ] )
            features["Pacman-Food-minDistance"] = minDistance
        except:
            print "All food have been eaten"
            pass

        try:
            minDistanceToCapsule = min( [ self.getMazeDistance( capsule, gameState.getAgentState( agentIndex ).getPosition() ) for capsule in gameState.getCapsules() ] )
            features[ "Pacman-Capsule-minDistance"] = minDistanceToCapsule
        except:
            print "Capsule has been eaten"
            pass

        ### Partition enemies into scared one and unscared one
        UnScaredEnemyList = []
        ScaredEnemyList = []
        for enemyIndex in self.enemies:
            if gameState.getAgentState( enemyIndex ).scaredTimer > 0:
                ScaredEnemyList.append( enemyIndex )
            else:
                UnScaredEenemyList.append( enemyIndex )

        ### Unscared ones
        if len( UnScaredEnemyList ) > 0:
            minDistanceToUnScaredEnemy = min( [ self.getMazeDistance( gameState.getAgentState( agentIndex ),  gameState.getMazeDistance( UnScaredEnemyIndex ) )
                                                for UnScaredEnemyIndex in UnScaredEnemyList ] )            
            features["Pacman-UnScaredEnemy-minDistance"] = minDistanceToUnScaredEnemy
            #features["Pacman-UnScaredEnemy-minDistance-numCarrying"] = gameState.getAgentState( agentIndex ).numCarrying

            InterceptList = self.isIntercept( gameState, [ agentIndex, ], UnScaredEnemyList, ObjsType = 0 )
            for index, ( agentIndex, interceptDistance, isIntercept ) in InerceptList:
                features["Pacman-UnScaredEnemy-flee-intercept-minDistance"] = interceptDistance
                features["Pacman-UnScaredEnemy-flee-isIntercept"] = int( isIntercept )

            InterceptList = self.isIntercept( gameState, [ agentIndex, ], UnScaredEnemyList, ObjsType = 1 )
            for index, ( agentIndex, interceptDistance, isIntercept ) in InerceptList:
                features["Pacman-UnScaredEnemy-capsule-intercept-minDistance"] = interceptDistance
                features["Pacman-UnScaredEnemy-capsule-isIntercept"] = int( isIntercept )

        if len( ScaredEnemyList ) > 0:
            ScaredTimer = gameState.getAgentState( ScaredEnemyList[0] ).scaredTimer
            InterceptList = self.isIntercept( gameState, self.allies, ScaredEnemyList, ObjsType = 0, scaredTimer = ScaredTimer )
            for index, ( agentIndex, isIntercept ) in InerceptList:
                #features["Pacman-UnScaredEnemy-flee-intercept-minDistance" + str(index)] = interceptDistance
                features["Pacman-ScaredEnemy-flee-isIntercept"] = int( isIntercept )
                #features["Pacman-ScaredEnemy-flee-numCarrying" + str(index) ] = gameState.getAgentState( Index ).numCarrying

        return features

    def getAgentGhostFeatures( self, gameState, agentIndex):
        features = util.Counter()
        ### Prepare  
        ### Partition the agents into two parts, scaring one and normal one 
        if gameState.getAgentState( agentIndex ).scaredTimer > 0:
            Scaring = True
        else:
            Scaring = False
        ### find that Invaders in our territory
        InvaderIndexList = [ agentIndex for agentIndex in self.enemies if gameState.getAgentState(agentIndex).isPacman ]
        ### scaring one
        if Scaring:
            DistanceToInvaderList = [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(), gameState.getAgentState( invaderIndex).getPosition() )
                                           for invaderIndex in InvaderIndexList ]
            minDistanceToInvader = min( DistanceToInvaderList ) 
            features["ScaringGhost-Invader-minDistance"] = minDistanceToInvader
            #minDistanceToInvaderIndex = minDistanceToInvaderList.index( minDistanceToInvader )
            #features["ScaringGhost-Invader-minDistance-numCarrying" + str( index ) ] = gameState.getAgentState( minDistanceToInvaderIndex ).numCarrying 
            minDistanceToEnemyField = min( [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(), Pos ) for Pos in self.EnemyBorder ] )
            features["ScaringGhost-EnenmyField-minDistance"] = minDistanceToEnemyField
            ### The following feature is not approproate! 
            InterceptList = self.isIntercept( gameState, InvaderIndexList, normalAgentIndexList, ObjsType = 0 )
            for index, ( InvaderIndex, interceptDistance, isIntercept) in enumerate(InterceptList):
		features["ScaringGhost-Invader-EnemyField-isIntercept"]

        else:
            InterceptList = self.isIntercept( gameState, InvaderIndexList, [ agentIndex, ], ObjsType = 0 )
            InterceptInfo = sorted( interceptList, key = lambda x: ( x[-1], x[-2] ) ,reverse = True )[0]
            features["NormalGhost-Invader-flee-intercept-distance"] = interceptInfo[1]
            features["NormalGhost-Invader-flee-isIntercept"] = interceptInfo[2]
            InterceptList = self.isIntercept( gameState, InvaderIndexList, [ agentIndex, ], ObjsType = 1 )
            InterceptInfo = sorted( InterceptList, key = lambda x: ( x[-1], x[-2] ), reverse = True )[0]
            #for index, ( InvaderIndex, interceptDistance, isIntercept) in enumerate(InterceptList):
            features["NormalGhost-Invader-capsule-intercept-distance"] = InterceptInfo[1]
            features["NormalGhost-Invader-capsule-isIntercept"] = InterceptInfo[2]
            
        return features

    def getWeights( self ):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        #return self.Param_Weights
        return {"stopped":-5, 
		"reverse":-5,
       		"eat-new-food":2,
      		"return-new-food":5,
		"return-new-food-number":2,
        	"eat-capsules":20, 
               	"Ally-Pacman-Die":-10,
                "Ally-Pacman-Die-food":-2,
		"Ally-Ghost-Die":-5,
              	"Enemy-Pacman-Die":10,
               	"Enemy-Pacman-Die-food":2,
		"Enemy-Ghost-Die":5,	
		"Pacman-Food-minDistance":-1,
                "Pacman-Capsule-minDistance":-2,
         	"Pacman-UnScaredEnemy-minDistance1":2,
                "Pacman-UnScaredEnemy-minDistance-numCarrying":2, 
                "Pacman-UnScaredEnemy-flee-intercept-minDistance":-2, 
                "Pacman-UnScaredEnemy-flee-isIntercept":10,
                "Pacman-UnScaredEnemy-capsule-intercept-minDistance":-2,
                "Pacman-UnScaredEnemy-capsule-isIntercept":5,
               	"Pacman-ScaredEnemy-flee-isIntercept1":5,
          	"ScaringGhost-Invader-minDistance":1,
            	"ScaringGhost-EnenmyField-minDistance":-2,
            	"NormalGhost-Invader-minDistance":-1,
                "NormalGhost-Invader-flee-intercept-distance":2,
                "NormalGhost-Invader-flee-isIntercept1":-10,
                "NormalGhost-Invader-capsule-intercept-distance":2,
              	"NormalGhost-Invader-capsule-isIntercept":-20,
                }


