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
from itertools import product, combinations, permutations


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
        print "x" * 50
        foodLeft = len(self.getFood( gameState ).asList())
        if foodLeft <= 2:
            bestDist = 9999
            for action in gameState.getLegalActions( self.index ):
                ### the following getSuccessor need to change
                successor = gameState.generateSuccessor( self.index, action)
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

        MaxMinScore = -999999
        MinScoreAlliesActionsList = []
        MinScoreList = []
        ChosedAlliesAction = None

        for AlliesAction in  list( product( list( product( [ self.allies[0], ], gameState.getLegalActions( self.allies[0] ) ) ),\
                                             list( product( [ self.allies[1], ], gameState.getLegalActions( self.allies[1] ) ) ) ) ):
            MinScore = 999999
            ChosedEnemiesAction = None
            for EnemiesAction in  list( product( list( product( [ self.enemies[0], ], gameState.getLegalActions( self.enemies[0] ) ) ),\
                                                 list( product( [ self.enemies[1], ], gameState.getLegalActions( self.enemies[1] ) ) ) ) ):

                IndexActions = AlliesAction + EnemiesAction
                #print IndexActions
                CurrentScore = self.evaluate( gameState, IndexActions )
                if CurrentScore < MinScore:
                    MinScore = CurrentScore
                    ChosedEnemiesAction = EnemiesAction

            MinScoreAlliesActionsList.append( ( AlliesAction, MinScore))
            MinScoreList.append( MinScore )

            #if MinScore!= 999999 and MinScore > MaxMinScore:
            #    MaxMinScore = MinScore
            #    ChosedAlliesAction = AlliesAction
        MaxMinScore = max( MinScoreList )
        BestActionList = []
        for AlliesAction, MinScore in MinScoreAlliesActionsList:
            if MinScore == MaxMinScore:
                BestActionList.append( AlliesAction )

        ChosedAlliesAction = random.sample( BestActionList, 1 )[0]
        global BestAction
        index = self.allies.index( self.index )
        for agentIndex, action in ChosedAlliesAction:
            if agentIndex == self.index:
                bestAction = action
            else:
                BestAction = action
        print "x" * 50
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
        agentMoveOrder = sorted( agentMoveInfo, key=lambda x:( x[0], x[1] ), reverse = True )

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
        #print "Current Positions:"
        #for agentIndex in self.enemies + self.allies:
        #    print gameState.getAgentState( agentIndex ).getPosition(),
        #print "\n"
        #print "actions:", actions
        #print "ActionFeatures:", ActionFeatures
        succGameState = self.getSuccessor( gameState, actions )
        #for agentIndex in self.enemies + self.allies:
        #    print succGameState.getAgentState( agentIndex ).getPosition()
        ActionScore = ActionFeatures * self.getWeights()
        #print "ActionScore:", ActionScore
        Score += ActionScore
        AgentFeaturesList = []
        for agentIndex in self.allies:
            AgentFeatures = self.getAgentFeatures( succGameState, agentIndex)
            #print "agentIndex:", agentIndex,"AgentFeatures:",AgentFeatures
            AgentFeaturesList.append( AgentFeatures )

        if AgentFeaturesList[0]["Pacman-UnScaredEnemy-capsule-isIntercept"] == 1 and \
            AgentFeaturesList[1]["Pacman-UnScaredEnemy-capsule-isIntercept"] == 1:
            AgentFeaturesList[0]["Pacman-UnScaredEnemy-capsule-isIntercept"] = 0

        for AgentFeatures in AgentFeaturesList:
            AgentScore = AgentFeatures * self.getWeights()
            Score += AgentScore
            #print AgentScore

        #print "=" * 50
        return Score

    def isIntercept( self, gameState, escapeAgentIndexList, chaseAgentIndexList, ObjsType = None, scaredTimer = -9999):
        ### First, we need to judge that whether the escapeAgentIndexList
        ### If
        if len( escapeAgentIndexList ) == 0 or len( chaseAgentIndexList ) == 0:
            return None
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
                Objs = gameState.getRedCapsules()
            else:
                Objs = gameState.getBlueCapsules()
        else:
            return None

        escapeAgentPositionList = [ gameState.getAgentState( agentIndex ).getPosition() for agentIndex in escapeAgentIndexList ]
        chaseAgentPositionList = [ gameState.getAgentState( agentIndex ).getPosition() for agentIndex in chaseAgentIndexList ]
        
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
                currentInterceptDistanceList = [ max( [ escapeAgentDistance - chaseAgentToBorderDistance[ PosIndex ], 2 * int( scaredTimer >= escapeAgentDistance ) - 1 ] )
                                          for chaseAgentToBorderDistance in chaseAgentToBorderDistanceList ]
                currentInterceptDistanceList.append(-1)
                interceptDistance.append( max( currentInterceptDistanceList ) )
            interceptDistanceList.append( min( interceptDistance ) )

        ### If distance is larger than zero, escapeAgents can be captured by chaseAgents!
        ### else the other condition is that the escapeAgent is able to escape!
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
        features["stopped"] = 0
        features["reverse"] = 0
        for agentIndex, action in actions:
            if agentIndex in self.allies:
                if action == Directions.STOP:
                    features["stopped"] += 1

                rev = Directions.REVERSE[ gameState.getAgentState(agentIndex).configuration.direction ]
                if rev == agentAction:
                    features["reverse"] += 1

        NewState = self.getSuccessor( gameState, actions )
        features["TeamDistance"] = self.getMazeDistance( NewState.getAgentState( self.allies[0] ).getPosition(),
                                                         NewState.getAgentState( self.allies[1] ).getPosition())
        features["eat-new-food"] = 0
        features["return-new-food"] = 0
        features["return-new-food-number"] = 0
        features["eat-capsules"] = 0
        for agentIndex in self.allies:
            OldAgentState = gameState.getAgentState( agentIndex )
            NewAgentState = NewState.getAgentState( agentIndex )
            features["eat-new-food"] += int( (NewAgentState.numCarrying - OldAgentState.numCarrying) > 0 )
            features["return-new-food"] += int( (NewAgentState.numReturned - OldAgentState.numReturned) > 0 )
            features["return-new-food-number"] += NewAgentState.numReturned - OldAgentState.numReturned
            features["eat-capsules"] += int( NewAgentState.numCapsules - OldAgentState.numCapsules < 0 )

        try:
            foodPositionList = self.getFood(gameState).asList()
            DistanceToFoodList = [ 0, ] * 2
            for index, agentIndex in enumerate(self.allies):
                agentPosition = NewState.getAgentState( agentIndex ).getPosition()
                DistanceToFoodList[index] = [self.getMazeDistance(foodPosition, agentPosition) for foodPosition in foodPositionList]

            minSumDistance = 999999
            d1 = None
            d2 = None
            for indexs in list( permutations( list(range(len( foodPositionList ))), 2 ) ):
                ix1, ix2 = indexs
                sumDistance = DistanceToFoodList[0][ix1] + DistanceToFoodList[1][ix2]
                if sumDistance < minSumDistance:
                    minSumDistance = minSumDistance
                    d1 = DistanceToFoodList[0][ix1]
                    d2 = DistanceToFoodList[1][ix2]

            features["Pacman-Food-minDistance"] = d1 + d2
        except:
            print "All food have been eaten"
            pass

        try:
            minMinDistanceToCapsule = 99999
            for agentIndex in self.allies:
                minDistanceToCapsule = min( [ self.getMazeDistance( capsule, gameState.getAgentState( agentIndex ).getPosition() )
                                          for capsule in self.getCapsules( gameState ) ] )
                if minMinDistanceToCapsule < minMinDistanceToCapsule:
                    minMinDistanceToCapsule = minDistanceToCapsule
            features[ "Pacman-Capsule-minDistance"] = minMinDistanceToCapsule
        except:
            print "There is no Capsule"
            pass

        ### CheckDeath
        Invaders = [ Index for Index in self.enemies if gameState.getAgentState( Index ).isPacman ]
        features["Ally-Pacman-Die"] = 0
        features["Ally-Pacman-Die-food"] = 0
        features["Ally-Ghost-Die"] = 0
        features["Shift-Pacman-Ghost"] = 0
        features["Shift-Ghost-Pacman"] = 0
        _, deathAgents = self.getSuccessor( gameState, actions, True )
        for agentIndex in self.allies:
            if agentIndex in deathAgents:
                OldAgentState = gameState.getAgentState( agentIndex )
                NewAgentState = gameState.getAgentState( agentIndex )
                if OldAgentState.isPacman and not NewAgentState.isPacman and len(Invaders) > 0:
                    features["Shift-Pacman-Ghost"] += 1
                if not OldAgentState.isPacman and NewAgentState.isPacman and len(Invaders) > 0:
                    features["Shift-Ghost-Pacman"] += 1

                if gameState.getAgentState( agentIndex ).isPacman:
                    features[ "Ally-Pacman-Die"] += 1
                    features[ "Ally-Pacman-Die-food" ] += gameState.getAgentState( agentIndex ).numCarrying
                else:
                    features[ "Ally-Ghost-Die"] += 1

        features["Enenmy-Pacman-Die"] = 0
        features["Enemy-Pacman-Die-food"] = 0
        features["Enemy-Ghost-Die"] = 0
        for agentIndex in self.enemies:
            if agentIndex in deathAgents:
                if gameState.getAgentState( agentIndex ).isPacman:
                    features[ "Enemy-Pacman-Die" ] += 1
                    features[ "Enemy-Pacman-Die-food" ] += gameState.getAgentState( agentIndex ).numCarrying
                else:
                    features["Enemy-Ghost-Die" ] += 1

        return features

    def getAgentFeatures( self, gameState, agentIndex):
        return self.getAgentPacmanFeatures( gameState, agentIndex) + self.getAgentGhostFeatures( gameState, agentIndex)

    def getAgentPacmanFeatures( self, gameState, agentIndex ):
        features = util.Counter()
        agentPosition = gameState.getAgentState( agentIndex ).getPosition()
        #print "agentIndex", agentIndex, "agentPosition",agentPosition
        ### Partition enemies into scared one and unscared one
        UnScaredEnemyList = []
        ScaredEnemyList = []
        for enemyIndex in self.enemies:
            if gameState.getAgentState( enemyIndex ).scaredTimer > 0:
                ScaredEnemyList.append( enemyIndex )
            else:
                UnScaredEnemyList.append( enemyIndex )

        ### Unscared ones
        if len( UnScaredEnemyList ) > 0:
            #minDistanceToUnScaredEnemy = min( [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(),  gameState.getAgentState( UnScaredEnemyIndex ).getPosition() )
            #                                    for UnScaredEnemyIndex in UnScaredEnemyList ] )
            #features["Pacman-UnScaredEnemy-minDistance"] = minDistanceToUnScaredEnemy
            #features["Pacman-UnScaredEnemy-minDistance-numCarrying"] = gameState.getAgentState( agentIndex ).numCarrying
            if gameState.getAgentState( agentIndex ).isPacman:
                InterceptList = self.isIntercept( gameState, [ agentIndex, ], UnScaredEnemyList, ObjsType = 0 )
                for agentIndex, interceptDistance, isIntercept in InterceptList:
                    features["Pacman-UnScaredEnemy-flee-intercept-minDistance"] = interceptDistance
                    features["Pacman-UnScaredEnemy-flee-isIntercept"] = int( isIntercept )

                UnScaredEnemyGhostList = []
                for agentIndex in UnScaredEnemyList:
                    if not gameState.getAgentState( agentIndex ).isPacman:
                        UnScaredEnemyGhostList.append( agentIndex )

                if len( UnScaredEnemyGhostList ) > 0:
                    InterceptList = self.isIntercept( gameState, [ agentIndex, ], UnScaredEnemyGhostList, ObjsType = 1 )
                    for agentIndex, interceptDistance, isIntercept in InterceptList:
                        features["Pacman-UnScaredEnemy-capsule-intercept-minDistance"] = interceptDistance
                        features["Pacman-UnScaredEnemy-capsule-isIntercept"] = int( isIntercept )

        if len( ScaredEnemyList ) > 0:
            ScaredTimer = gameState.getAgentState( ScaredEnemyList[0] ).scaredTimer
            InterceptList = self.isIntercept( gameState, [ agentIndex, ], ScaredEnemyList, ObjsType = 0, scaredTimer = ScaredTimer )
            #print "InterceptList 352",InterceptList
            #for agentIndex, isIntercept in InterceptList:
                #features["Pacman-UnScaredEnemy-flee-intercept-minDistance" + str(index)] = interceptDistance
            features["Pacman-ScaredEnemy-flee-isIntercept"] = int( InterceptList[-1][-1] )
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
        InvaderIndexList = [ index for index in self.enemies if gameState.getAgentState( index ).isPacman ]
        ### scaring one
        if Scaring:
            if not gameState.getAgentState(agentIndex).isPacman:
                DistanceToInvaderList = [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(), gameState.getAgentState( invaderIndex).getPosition() )
                                               for invaderIndex in InvaderIndexList ]
                minDistanceToInvader = min( DistanceToInvaderList )
                features["ScaringGhost-Invader-minDistance"] = minDistanceToInvader
                #minDistanceToInvaderIndex = minDistanceToInvaderList.index( minDistanceToInvader )
                #features["ScaringGhost-Invader-minDistance-numCarrying" + str( index ) ] = gameState.getAgentState( minDistanceToInvaderIndex ).numCarrying
                minDistanceToEnemyField = min( [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(), Pos ) for Pos in self.EnemyBorder ] )
                features["ScaringGhost-EnenmyField-minDistance"] = minDistanceToEnemyField
                ### The following feature is not approproate!
                InterceptList = self.isIntercept( gameState, InvaderIndexList, [ agentIndex, ], ObjsType = 0 )
                for index, ( InvaderIndex, interceptDistance, isIntercept) in enumerate(InterceptList):
                    features["ScaringGhost-Invader-EnemyField-isIntercept"]

        else:
            if gameState.getAgentState( agentIndex ).isPacman:
                DistanceToInvaderList = [self.getMazeDistance(gameState.getAgentState(agentIndex).getPosition(),
                                                              gameState.getAgentState(invaderIndex).getPosition())
                                         for invaderIndex in InvaderIndexList]
                #if len(DistanceToInvaderList) > 0:
                #    print DistanceToInvaderList
                    # raise Exception
                try:
                    minDistanceToInvader = min(DistanceToInvaderList)
                    features["NormalPacman-Invader-minDistance"] = minDistanceToInvader
                except:
                    #print "Invaders is None"
                    pass

            else:
            #if not gameState.getAgentState( agentIndex ).isPacman:
                #try:
                DistanceToInvaderList = [ self.getMazeDistance( gameState.getAgentState( agentIndex ).getPosition(), gameState.getAgentState( invaderIndex).getPosition() )
                                               for invaderIndex in InvaderIndexList ]
                #if len(DistanceToInvaderList) > 0:
                #    print DistanceToInvaderList
                    #raise Exception
                try:
                    minDistanceToInvader = min( DistanceToInvaderList )
                    features["NormalGhost-Invader-minDistance"] = minDistanceToInvader
                except:
                    #print "Invaders is None"
                    pass

                InterceptList = self.isIntercept(gameState, InvaderIndexList, [agentIndex, ], ObjsType=1)
                if InterceptList is not None and len(InterceptList) > 0:
                    InterceptInfo = sorted(InterceptList, key=lambda x: (x[-1], x[-2]), reverse=True)[0]
                    # for index, ( InvaderIndex, interceptDistance, isIntercept) in enumerate(InterceptList):
                    features["NormalGhost-Invader-capsule-intercept-distance"] = InterceptInfo[1]
                    features["NormalGhost-Invader-capsule-isIntercept"] = InterceptInfo[2]

            InterceptList = self.isIntercept( gameState, InvaderIndexList, [ agentIndex, ], ObjsType = 0 )
            if InterceptList is not None and len(InterceptList) > 0:
                InterceptInfo = sorted( InterceptList, key = lambda x: ( x[-1], x[-2] ) ,reverse = True )[0]
                features["NormalGhost-Invader-flee-intercept-distance"] = InterceptInfo[1]
                features["NormalGhost-Invader-flee-isIntercept"] = InterceptInfo[2]
            
        return features

    def getWeights( self ):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        #return self.Param_Weights
        return {"stopped":-20,
		    "reverse":-20,

            "TeamDistance":0,
       		"eat-new-food":10,
      		"return-new-food":40,
		    "return-new-food-number":1,
        	"eat-capsules":60,

            "Ally-Pacman-Die":-100000,
            "Ally-Pacman-Die-food":-1000,
		    "Ally-Ghost-Die":-100000,
            "Enemy-Pacman-Die":500,
            "Enemy-Pacman-Die-food":5,
		    "Enemy-Ghost-Die":500,

            "Shift-Pacman-Ghost":20,
            "Shift-Ghost-Pacman":-20,

            "NormalPacman-Invader-minDistance":-0.5,
		    "Pacman-Food-minDistance":-1,
            "Pacman-Capsule-minDistance":-8,
         	"Pacman-UnScaredEnemy-minDistance":0,
            "Pacman-UnScaredEnemy-minDistance-numCarrying":0,
            "Pacman-UnScaredEnemy-flee-intercept-minDistance":0,
            "Pacman-UnScaredEnemy-flee-isIntercept":-20,#,100,
            "Pacman-UnScaredEnemy-capsule-intercept-minDistance":0,#-20,
            "Pacman-UnScaredEnemy-capsule-isIntercept":0,#-200,
            "Pacman-ScaredEnemy-flee-isIntercept":0,
          	"ScaringGhost-Invader-minDistance":0,
            "ScaringGhost-EnenmyField-minDistance":0,
            "NormalGhost-Invader-minDistance":-20,#10,
            "NormalGhost-Invader-flee-intercept-distance":0,#2,
            "NormalGhost-Invader-flee-isIntercept":20,#100,
            "NormalGhost-Invader-capsule-intercept-distance":0,#4,
            "NormalGhost-Invader-capsule-isIntercept":30,#200,
            }


