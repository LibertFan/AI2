from capture import runGames, readCommand
import random, copy
from pathos import multiprocessing as mp

def get_basic_command_argv():
    return BasicCommandArgv = { 'redOpts': '', 'blueOpts': '', 'catchExceptions': False, 'blue_name': 'Blue', 'numTraining': 0, 'replay': None, 'super_quiet': False, 'blue': 'baselineTeam', 'red_name': 'Red', 'layout': 'defaultCapture', 'numGames': 1, 'red': 'baselineTeam', 'textgraphics': False, 'fixRandomSeed': True, 'keys1': False, 'keys0': False, 'keys3': False, 'keys2': False, 'quiet': True, 'record': False, 'time': 1200, 'zoom': 1, 'Param_Weights_1': None, 'Param_Weights_2': None}

def GameContext:
    def __init__( self, alpha = alpha, NumUnit = 30, EnemiesAgentList = None, BasicAgent = None ):
        self.BasicAgent = BasicAgent
        self.EnemiesAgentList = EnemiesAgentList
        if self.EnemiesAgentList is None: 
           self.EnemeisAgentList = [ " ", ]          
          
        self.NumUnit = NumUnit
        self.alpha = alpha
        
        self.P = 
        self.Options = None
        self.Params = None
        self.Exploits = None         
  
    def set_initial_weights( self,  =   ):
        pass
  
    def 

def Evaluate():
	pass

def Champion():
	pass

def CrossOver():
	pass

def mutation():
	pass




def main():
	random.seed(10)
	options = readCommand(sys.argv[1:])  # Get game components based on input
	options_list = [ options , copy.deepcopy(options), copy.deepcopy(options) ]
	p = mp.ProcessPool( 4 )
	t1 = time.time()
		                        
	ActionSeriesLists = []
	results = []
	for o in options_list:
		results.append( p.apipe( runGames, **o  ) ) 
	for r in results:
		ActionSeriesLists.append( r.get() )
	for v in ActionSeriesLists:
		print v


if __name__=="__main__":
    main()

