from capture import runGames, readCommand, MP
import random, copy
from pathos import multiprocessing as mp
import numpy as np
#import multiprocessing as mp
NumGames = 40
class Options(object):
    def __init__( self, redOpts = '', blueOpts = '', catchExceptions = False, blue_name = 'Blue', \
                  numTraining = 0, replay = None, super_quiet = False,blue = 'baselineTeam', \
                  red_name = 'Red', layout = 'defaultCapture', numGames = NumGames, red = 'baselineTeam',\
                  textgraphics = False, fixRandomSeed = False, keys1 = False, keys0 = False, keys3 = False,\
                  keys2 = False, quiet = True, record = False, time = 1200, zoom = 1, Param_Weights_1 = None,\
                  Param_Weights_2 = None, serial_num = None ):
        self.redOpts = redOpts
        self.blueOpts = blueOpts
        self.catchExceptions = catchExceptions
        self.blue_name = blue_name
        self.numTraining = numTraining 
        self.replay = replay
        self.super_quiet = super_quiet
        self.blue = blue
        self.red_name = red_name
        self.layout = layout
        self.numGames = numGames
        self.red = red
        self.textgraphics = textgraphics
        self.fixRandomSeed = fixRandomSeed
        self.keys1 = keys1
        self.keys0 = keys0
        self.keys3 = keys3
        self.keys2 = keys2
        self.quiet = quiet
        self.record = record
        self.time = time
        self.zoom = zoom 
        self.Param_Weights_1 = Param_Weights_1
        self.Param_Weights_2 = Param_Weights_2
        #self.Record = ( str( self.red_name), str( self.blue_name), 0, 0, 0 ) 
        self.serial_num = serial_num 

class EvolutionAlgorithm(object):
    def __init__( self, alpha = 0.6, NumUnit = 30, EnemiesAgentList = None, BasicAgent = None ):
        self.BasicAgent = BasicAgent
        self.EnemiesAgentList = EnemiesAgentList
        if self.EnemiesAgentList is None: 
            #self.EnemiesAgentList = [ "enemy3" , "myTeam_caesar", "enemy1", "enemy2", ]          
            self.EnemiesAgentList = [ "baselineTeam", ]          
         
        self.NumUnit = NumUnit
        self.alpha = alpha
        self.evolution_generations = 200                 

    def set_initial_weights( self ):
        agent_weights_and_serial_info_dict = dict() 
        #initial_command_list = []
        #agent_weights_dict = dict()
        for i in range( self.NumUnit ):
            initial_weights = dict()
            initial_weights["stopped"] = round( random.random()*2, 2 ) * -5, 
            initial_weights["reverse"] = round( random.random()*2, 2 ) * -5,
       	    initial_weights["eat-new-food"] = round( random.random()*2, 2 ) * 2,
      	    initial_weights["return-new-food"] = round( random.random()*2, 2) * 5,
	    initial_weights["return-new-food-number"] = round( random.random()*2, 2 ) * 2,
            initial_weights["eat-capsules"] = round( random.random()*2, 2 ) * 20,
            initial_weights["Ally-Pacman-Die"] = round( random.random()*2, 2 ) * -10,
            initial_weights["Ally-Pacman-Die-food"] = round( random.random()*2, 2 ) * -2,
	    initial_weights["Ally-Ghost-Die"] = round( random.random()*2, 2 ) * -5,
            initial_weights["Enemy-Pacman-Die"] = round( random.random()*2, 2 ) * 10,
            initial_weights["Enemy-Pacman-Die-food"] = round( random.random()*2, 2 ) * 2,
	    initial_weights["Enemy-Ghost-Die"] = round( random.random()*2, 2 ) * 5,	
	    initial_weights["Pacman-Food-minDistance1"] = round( random.random()*2, 2 ) * -1,
	    initial_weights["Pacman-Food-minDistance2"] = round( random.random()*2, 2 ) * -1,
            initial_weights["Pacman-Capsule-minDistance1"] = round( random.random()*2, 2 ) * -2,
	    initial_weights["Pacman-Capsule-minDistance2"] = round( random.random()*2, 2 ) * -2,
            initial_weights["Pacman-UnScaredEnemy-minDistance1"] = round( random.random()*2, 2 ) * 2,
            initial_weights["Pacman-UnScaredEnemy-minDistance-numCarrying1"] = round( random.random()*2, 2 ) * 2, 
            initial_weights["Pacman-UnScaredEnemy-flee-intercept-minDistance1"] = round( random.random()*2, 2 ) * -2, 
            initial_weights["Pacman-UnScaredEnemy-flee-isIntercept1"] = round( random.random()*2, 2 ) * 10,
            #"Pacman-UnScaredEnemy-flee-numCarrying1":,
            initial_weights["Pacman-UnScaredEnemy-capsule-intercept-minDistance1"] = round( random.random()*2, 2 ) * -2,
            initial_weights["Pacman-UnScaredEnemy-capsule-isIntercept1"] = round( random.random()*2, 2 ) * 5,
            initial_weights["Pacman-ScaredEnemy-flee-isIntercept1"] = round( random.random()*2, 2 ) *
            #"Pacman-ScaredEnemy-flee-numCarrying1":,
            initial_weights["Pacman-UnScaredEnemy-minDistance2"] ,
                "Pacman-UnScaredEnemy-minDistance-numCarrying2" 
                "Pacman-UnScaredEnemy-flee-intercept-minDistance2" 
                "Pacman-UnScaredEnemy-flee-isIntercept2"
                "Pacman-UnScaredEnemy-flee-numCarrying2"
                "Pacman-UnScaredEnemy-capsule-intercept-minDistance2"
                "Pacman-UnScaredEnemy-capsule-isIntercept2"
               	"Pacman-ScaredEnemy-flee-isIntercept2"
              	"Pacman-ScaredEnemy-flee-numCarrying2"
            initial_weights["ScaringGhost-Invader-minDistance1"] = round( random.random()*2, 2 ) * 1,
            	#"ScaringGhost-Invader-minDistance-numCarrying1":, 
            initial_weights["ScaringGhost-EnenmyField-minDistance1"] = round( random.random()*2, 2) * -2,
		###"ScaringGhost-Invader-EnemyField-isIntercept":-5,				

            initial_weights["NormalGhost-Invader-minDistance1"] = round( random.random()*2,2) * -1,
		#"NormalGhost-Invader-minDistance-numCarrying1":-0.1,
            initial_weights["NormalGhost-Invader-flee-intercept-distance1": round( random.random()*2, 2) * 2,
            initial_weights["NormalGhost-Invader-flee-isIntercept1"] = round( random.random()*2, 2) * -10
                #"NormalGhost-Invader-flee-intercept-numCarrying1":,
            initial_weights["NormalGhost-Invader-capsule-intercept-distance1"] = round( random.random() *2, 2) * 2,
            initial_weights["NormalGhost-Invader-capsule-isIntercept1"] = round( random.random() *2, 2) * -20
          	"ScaringGhost-Invader-minDistance2"
            	"ScaringGhost-Invader-minDistance-numCarrying2" 
            	"ScaringGhost-EnenmyField-minDistance2"
            	"NormalGhost-Invader-minDistance2"
		"NormalGhost-Invader-minDistance-numCarrying2"
                "NormalGhost-Invader-flee-intercept-distance2"
                "NormalGhost-Invader-flee-isIntercept2"
                "NormalGhost-Invader-flee-intercept-numCarrying2" 
                "NormalGhost-Invader-capsule-intercept-distance2"
              	"NormalGhost-Invader-capsule-isIntercept2"
        
        return agent_weights_and_serial_info_dict
    
    def convert_weights_to_option( self, agent_weights_and_serial_info_dict):
        argv_list = []
        for agent_weights, serial_info in agent_weights_and_serial_info_dict.values():
            weights_1, weights_2 = agent_weights 
            for index, EnemiesAgent in enumerate( self.EnemiesAgentList ):
		argv = Options(blue=self.BasicAgent, red=EnemiesAgent, Param_Weights_1=weights_1, Param_Weights_2=weights_2, serial_num = serial_info )
		argv_list.append( argv )
        return argv_list 

    ### the output of evaluate should be paired withe argv_list
    def evaluate( self, argv_list ):
        score_dict = dict()
        print "c"*50
        p = mp.ProcessPool( 24 )
        results = []
        for argv in argv_list:
            #print "argv", argv
            results.append( p.apipe( MP, argv ) )
 
        for r in results:
            c = r.get() 
            scores, redWinRate, blueWinRate, current_serial_info = c
            print "=" * 50
            print scores, redWinRate, blueWinRate
            print "current_serial_info"
            print current_serial_info  
            agent_index, CurrentGameNum, CurrentAverageScore, CurrentWinRate = current_serial_info
            if score_dict.get( agent_index ) is None:
               score_dict[ agent_index ] = ( agent_index, CurrentGameNum, CurrentAverageScore, CurrentWinRate )
            agent_index, CurrentGameNum, CurrentAverageScore, CurrentWinRate = score_dict.get( agent_index )
                        
            SumGameNum = NumGames + CurrentGameNum
            ### the following set if specific for red team
            # SumScore = CurrentGameNum * CurrentAverageScore + sum( scores )
            # SumWin = CurrentGameNum * CurrentWinRate + NumGames * redWinRate

            ### the following set is specfic for blue team 
            SumScore = CurrentGameNum * CurrentAverageScore - sum( scores )
            SumWin = CurrentGameNum * CurrentWinRate + NumGames * blueWinRate
            current_serial_info = ( agent_index, SumGameNum, SumScore / float( SumGameNum), SumWin / float( SumGameNum ) )  
            score_dict[ agent_index ] =  current_serial_info
            
            score_mat = np.zeros( [ self.NumUnit, 4 ] )
            for index in range(len(score_mat)):
                score_mat[ index ] = score_dict.get( index )
 
        return score_mat
       
    def champion( self, score_mat, agent_weights_and_serial_info_dict ):        	
        #agent_WinRate_Score_pair_list = list( zip( list( range( len( agent_WinRate_Score_pair_list ) ) ), agent_WinRate_Score_pair_list ) )
        #print "selected_agent_score_pair_list", selected_agent_WinRate_Score_pair_list
        selected_sorted_agent_weights_and_serial_info_list = []
        sorted_score_mat = np.array( sorted( score_mat, key = lambda x: ( x[-1], x[-2] ), reverse = True ) )
        for info in sorted_score_mat[ : self.NumUnit / 2 ]:
            agent_index = info[0]
            agent_weights, serial_info = agent_weights_and_serial_info_dict[ agent_index ] 
            new_agent_weights_and_serial_info = ( agent_weights, info )
 
            selected_sorted_agent_weights_and_serial_info_list.append( new_agent_weights_and_serial_info )

        return selected_sorted_agent_weights_and_serial_info_list
   
    def crossover( self, agent_weights1, agent_weights2 ): 
        agent_weights_list = [ agent_weights1, agent_weights2 ]  
        new_weights_1 = dict()
        for k in agent_weights1[0].keys():
            index = random.randint(0,1)
            new_weights_1[k] = agent_weights_list[index][0][k] 
        new_weights_2 = dict() 
        for k in agent_weights1[1].keys():
            index = random.randint(0,1)
            new_weights_2[k] = agent_weights_list[index][1][k]  
        return ( new_weights_1, new_weights_2 )
   
    def mutations( self, agent_weights, mu=0, sigma = 1 ):
        new_weights_1 = dict()
        for k, v in agent_weights[0].items():
            v += random.gauss( mu, sigma ) 
            new_weights_1[k] = v
        new_weights_2 = dict()
        for k, v in agent_weights[1].items():
            v += random.gauss( mu, sigma ) 
            new_weights_2[k] = v

        return ( new_weights_1, new_weights_2 ) 

    def GenerateAgents( self, agent_weights_and_serial_info_list, mu=0, sigma = 1):
        if len( agent_weights_and_serial_info_list ) != 5:
            raise Exception( " Left weights is not equal to half the NumUnit")
        else:
            new_weights_and_serial_info_list = []
            for weights, serial_info in agent_weights_and_serial_info_list:
                for i in range(4):
                    new_weights = self.mutations( weights )
                    new_weights_and_serial_info_list.append( ( new_weights, ( -1, 0, 0, 0 ) ) )
             
            pair_weights_and_serial_info_list = list( itertools.permutations( agent_weights_and_serial_info_list, 2 ) )
            for a1, a2 in pair_weights_and_serial_info_list:
                weights_1, _ = a1
                weights_2, _ = a2
                for i in range(2):
                    new_weights = self.crossover( weights_1, weights_2 )
                    new_weights_and_serial_info_list.append( ( new_weights, ( -1, 0, 0, 0) ) )

            all_weights_and_serial_info_list = agent_weights_and_serial_info_list + new_weights_and_serial_info_list    
            agent_weights_dict = dict()
            for index, agent_weights_and_serial_info in enumerate( all_weights_and_serial_info_list ):
                agent_weights, serial_info = agent_weights_and_serial_info 
                _, a, b, c = serial_info 
                new_serial_info = ( index, a, b, c )
                agent_weights_dict[index] = ( agent_weights, new_serial_info )
     
 	    return agent_weights_dict 
          

 
    def mutation( self, agent_weights_and_serial_info_list, mu=0, sigma=1 ):
        if len( agent_weights_and_serial_info_list ) != self.NumUnit / 2:
            raise Exception( " Left weights is not equal to half the NumUnit")
 	else:
            new_weights_and_serial_info_list = [] 
            index = self.NumUnit / 2
            for weights, serial_info in agent_weights_and_serial_info_list:
                new_weights_1 = dict()
                for k, v in weights[0].items():
                    v += random.gauss( mu, sigma ) 
                    new_weights_1[k] = v
                new_weights_2 = dict()
                for k, v in weights[1].items():
                    v += random.gauss( mu, sigma ) 
                    new_weights_2[k] = v
                new_weights = ( new_weights_1, new_weights_2 )
                new_weights_and_serial_info_list.append( ( new_weights, ( index, 0, 0, 0 ) ) )
                index += 1
            
            all_weights_and_serial_info_list = agent_weights_and_serial_info_list + new_weights_and_serial_info_list    
            agent_weights_dict = dict()
            for index, agent_weights_and_serial_info in enumerate( all_weights_and_serial_info_list ):
                agent_weights, serial_info = agent_weights_and_serial_info 
                _, a, b, c = serial_info 
                new_serial_info = ( index, a, b, c )
                agent_weights_dict[index] = ( agent_weights, new_serial_info )
     
 	    return agent_weights_dict 

    def evolution( self ):
        agent_weights_dict = self.set_initial_weights()
        store_scores = []
        for iters in range( self.evolution_generations ):
            argv_list = self.convert_weights_to_option( agent_weights_dict )
            #agent_score_pair_list, score_mat, redWinRateList, blueWinRateList, serial_num = self.evaluate( argv_list )
            score_mat = self.evaluate( argv_list )

            store_scores.append( score_mat )
            selected_sorted_weights_list = self.champion( score_mat, agent_weights_dict )
            agent_weights_dict = self.mutation( selected_sorted_weights_list ) 
             
            with open("score_0623v4_rB.txt","a") as f:
                 f.write( str( iters ) )
                 f.write( "\n" )
                 f.write( str( score_mat ) )
                 #f.write( "\n" )
                 #f.write( str( redWinRateList ) )
                 #f.write( "\n" ) 
                 #f.write( str( blueWinRateList ) )                 
                 f.write( "\n" ) 
                 f.write( str( agent_weights_dict[0] ) ) 
                 f.write( "\n\n\n" )

            f.close() 

        return agent_weights_dict, np.array( store_scores )

    def RoundRobin( self ):
        return 0            

    def set_weights( self ):
        return 0 


def main():
    ea = EvolutionAlgorithm( BasicAgent = "myTeamBasic" )
    agent_weights_dict, store_scores = ea.evolution() 
    with open("agent_weights_dict_0623v4_rB.txt","w") as f:
         f.write( str(agent_weights_dict) )
         f.write("\n\n\n")
         f.write( str(store_scores) )
    f.close() 
    np.save( "score_0623v4_rB.npy", store_scores )

if __name__=="__main__":
    main()

