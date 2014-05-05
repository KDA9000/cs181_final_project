from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import pickle
from sklearn.externals import joblib
import sys
from util import manhattanDistance

class BaseStudentAgent(object):
    """Superclass of agents students will write"""

    def registerInitialState(self, gameState):
        """Initializes some helper modules"""
        import __main__
        self.display = __main__._display
        self.distancer = Distancer(gameState.data.layout, False)
        self.firstMove = True

    def observationFunction(self, gameState):
        """ maps true state to observed state """
        return ObservedState(gameState)

    def getAction(self, observedState):
        """ returns action chosen by agent"""
        return self.chooseAction(observedState)

    def chooseAction(self, observedState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP
        
## Below is the class students need to rename and modify

class ExampleTeamAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(ExampleTeamAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        
    
    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition())
                              for gs in ghost_states])
        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action

class DataCollectorAgent(BaseStudentAgent):
    """
    Collects data for the game
    """
    data_points = 0
    max_points = 500000
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(DataCollectorAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")

    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        Move towards the closest capsule, ignore everything else
        """
        if observedState.scaredGhostPresent():
            sys.stderr.write("Real!\n")
            return Directions.STOP

        if self.data_points == self.max_points:
            print("Maximum data points collected!")
            sys.stderr.write("Maximum data points collected!\n")
            sys.exit(1)

        print(observedState.getGoodCapsuleExamples())
        pacmanPosition = observedState.getPacmanPosition()
        capsules = observedState.getCapsuleData()
        legal_moves = observedState.getLegalPacmanActions()
        #capsule_dist = np.array([self.distancer.getDistance(pacmanPosition,
        #    cap) for cap in capsule_locs])
        # list of ((x,y),d), where (x,y) is the location of the capsule, and d
        # is the distance from your current position to the capsule
        #closest = sorted(zip(capsule_locs, capsule_dist), key =
        #        lambda t: t[1])
        best_action = Directions.STOP
        best_dist = np.inf
        index = -1
        for move in legal_moves:
            if move == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition, move)
            dist = [self.distancer.getDistance(successor_pos, cap[0]) for cap in
                    capsules]
            new_index = np.argmin(dist)
            new_dist = dist[new_index]
            if new_dist < best_dist:
                best_action = move
                best_dist = new_dist
        if best_dist == 0:
            self.data_points += 1
            capsule_feat = capsules[index][1]
            out = str(capsule_feat[0]) + "," + str(capsule_feat[1]) + "," + str(capsule_feat[2])
            sys.stderr.write(out)
            sys.stderr.write("\n")

        return best_action

class heuristicsAgent(BaseStudentAgent):
    '''
    Our actual agent
    '''
    ghostsdict = None
    capsules = None
    prev_state = None
    prev_action = None
    optimal_action = None
    prev_score = 0
    clfGhost = None
    class_to_value = [29.9735,49.3226,150.649,19.9242]
    dirs = [Directions.NORTH,Directions.EAST,Directions.SOUTH,Directions.WEST,Directions.STOP]
    trapped_dir = None

    # stuff for doing k-means on capsules
    norm = None
    km = None
    correct_class = None

    def ddist_ghost(self,state,action):
        pacState = state.getPacmanState()
        legalActions = state.getLegalPacmanActions()
        pacpos = pacState.getPosition()
        speed = 1.0

        actionVector = Actions.directionToVector( action , speed )
        ghostsPositions = [self.ghostsdict['bad'][0].getPosition()]
        for i in range(3):
            ghostsPositions.append(self.ghostsdict['good'][i][0].getPosition())
        newPosition = (pacpos[0]+actionVector[0], pacpos[1]+actionVector[1])
        curDistancesG = [manhattanDistance(pacpos,ghostPosition) for ghostPosition in ghostsPositions]
        distanceGhosts = [manhattanDistance(newPosition,ghostPosition) for ghostPosition in ghostsPositions]


        # feature vectors for capsules
        capsulesPositions = [cap[0] for cap in self.capsules]
        curDistancesC = [manhattanDistance(pacpos,capsulePosition) for capsulePosition in capsulesPositions]
        distanceCapsules = [manhattanDistance(newPosition,capsulePosition) for capsulePosition in capsulesPositions]

        distances = np.array(distanceGhosts+distanceCapsules)
        curDistances = np.array(curDistancesG+curDistancesC)

        candiAns = (distances-curDistances)/np.absolute(curDistances.clip(1))
        if state.scaredGhostPresent():
            candiAns[0] = -candiAns[0]
        return candiAns

    def __init__(self, *args, **kwargs):
        pass

    def registerInitialState(self, gameState):
        super(heuristicsAgent, self).registerInitialState(gameState)
        with open('SVM_multi_linear_size_10000_011','rb') as fp:
            self.clfGhost = pickle.load(fp)
        with open('normalization_params', 'rb') as fp2:
            self.norm = pickle.load(fp2)
        with open('kmeans_params', 'rb') as fp3:
            self.km = pickle.load(fp3)


    def classifyGhost(self, feat_v, quad):
        clf_v = np.insert(feat_v,0,quad)
        return int(self.clfGhost.predict(clf_v)[0])

    # returns bool whether it's likely to be a good capsule or not
    # considered good capsule if it's in the same cluster as the example capsules
    def classifyCapsule(self, feat_v):
        # only get correct class once
        if self.correct_class == None:
            real_caps = np.array([[0.57992318,1.32338916,0.94076627],
                [-1.14773678,-4.82263876,1.21043419],
                [-0.04505671,-1.27733715,-0.11816285],
                [-2.43469675,-1.13362012,1.39733327],
                [-0.45958987,-1.02964365,0.63902567]])
            real_caps = self.norm.transform(real_caps)
            res = self.km.predict(real_caps)
            print("predicted classes for good capsule examples:")
            print(res)
            self.correct_class = int(np.mean(res) + 0.5)

        normed_feats = self.norm.transform(feat_v)
        prediction = int(self.km.predict(normed_feats))
        return (prediction == self.correct_class)



    def get_good_caps(self, observedState):
        caps_and_class = [(cap, self.classifyCapsule(cap[1])) for cap in observedState.getCapsuleData()]
        good_caps = [cap_class[0] for cap_class in caps_and_class if cap_class[1]]
        sorted_good_caps = sorted(good_caps, key = lambda x: manhattanDistance(x[0], observedState.getPacmanPosition()))
        return sorted_good_caps

        

    def chooseAction(self, observedState):
        pacmanPosition = observedState.getPacmanPosition()

        # if ghosts not initialized, initialize it to all ghosts currently on screen
        if self.ghostsdict == None:
            ghost_states = observedState.getGhostStates()
            self.ghostsdict = {'bad':None,'good':[]}
            for gs in ghost_states:
                clas = self.classifyGhost(gs.getFeatures(),observedState.getGhostQuadrant(gs))
                if clas == 5:
                    self.ghostsdict['bad'] = (gs, observedState.getGhostQuadrant(gs),-1000)
                else:
                    self.ghostsdict['good'].append((gs, observedState.getGhostQuadrant(gs), self.class_to_value[clas])) 

        # check at every step whether ghosts have changed by checking feature vectors
        else:
            new_ghost_states = observedState.getGhostStates()
            new_good_ghosts = []
            
            for new_gs in new_ghost_states:
                #clas = self.classifyGhost(new_gs.getFeatures(), observedState.getGhostQuadrant(new_gs))
                if (new_gs.getFeatures() == self.ghostsdict['bad'][0].getFeatures()).all():
                    self.ghostsdict['bad'] = (new_gs,self.ghostsdict['bad'][1],self.ghostsdict['bad'][2])
                else:
                    is_new = True
                    for gs_quad in self.ghostsdict['good']:
                        if (new_gs.getFeatures() == gs_quad[0].getFeatures()).all():
                            new_good_ghosts.append((new_gs,gs_quad[1],gs_quad[2]))
                            is_new = False
                            break
                    if is_new:
                        clas = self.classifyGhost(new_gs.getFeatures(), observedState.getGhostQuadrant(new_gs))
                        if clas == 5:
                            self.ghostsdict['bad'] = (new_gs, observedState.getGhostQuadrant(new_gs), -1000.)
                        else:
                            new_good_ghosts.append((new_gs, observedState.getGhostQuadrant(new_gs), self.class_to_value[clas]))
        

        bad_ghost_pos = self.ghostsdict['bad'][0].getPosition()


        # go towards scared ghost if present
        if observedState.scaredGhostPresent():
            legal_actions = observedState.getLegalPacmanActions()
            best_action = Directions.STOP
            best_dist = np.inf
            for move in legal_actions:
                if move == Directions.STOP:
                    continue
                successor_pos = Actions.getSuccessor(pacmanPosition, move)
                dist = manhattanDistance(successor_pos, bad_ghost_pos)
                if dist < best_dist:
                    best_action = move
                    best_dist = dist
            assert(best_action != Directions.STOP)
            self.prev_action = best_action
            return best_action
        # go towards good capsule unless move will immediately endanger ghost
        else:
            legal_actions = observedState.getLegalPacmanActions()
            good_caps = self.get_good_caps(observedState)
            dest = good_caps[0][0]
            best_action = Directions.STOP
            best_dist_to_cap = manhattanDistance(pacmanPosition, dest)
            for move in legal_actions:
                if move == Directions.STOP:
                    continue
                successor_pos = Actions.getSuccessor(pacmanPosition, move)
                print(pacmanPosition, successor_pos, move)
                dist_to_cap = manhattanDistance(successor_pos, dest)
                dist_to_bad_ghost = manhattanDistance(successor_pos, bad_ghost_pos)
                if dist_to_cap < best_dist_to_cap and dist_to_bad_ghost > 0:
                    best_action = move
                    best_dist_to_cap = dist_to_cap
                self.prev_action = best_action
            return best_action
                    


class KDA9000Agent(BaseStudentAgent):
    '''
    Our actual agent
    '''
    ghostsdict = None
    capsules = None
    class_to_value = [29.9735,49.3226,150.649,19.9242]

    J = 10
    # thetas = np.zeros(J)

    thetas = [120.072337981216242, -0.48824095638567144, -0.12732825706817857, -14.119137086538544] + [0.]*6
    prev_state = None
    prev_action = None
    optimal_action = None
    prev_score = 0
    clfGhost = None
    alpha = 0.0005
    gamma = 1-pow(10,-3)

    dirs = [Directions.NORTH,Directions.EAST,Directions.SOUTH,Directions.WEST,Directions.STOP]
    t = 3
    dt = 0.005
    trapped_dir = None

    # stuff for doing k-means on capsules
    norm = None
    km = None
    correct_class = None

    def ddist_ghost(self,state,action):
        pacState = state.getPacmanState()
        legalActions = state.getLegalPacmanActions()
        pacpos = pacState.getPosition()
        speed = 1.0

        actionVector = Actions.directionToVector( action , speed )
        ghostsPositions = [self.ghostsdict['bad'][0].getPosition()]
        for i in range(3):
            ghostsPositions.append(self.ghostsdict['good'][i][0].getPosition())
        newPosition = (pacpos[0]+actionVector[0], pacpos[1]+actionVector[1])
        curDistancesG = [manhattanDistance(pacpos,ghostPosition) for ghostPosition in ghostsPositions]
        distanceGhosts = [manhattanDistance(newPosition,ghostPosition) for ghostPosition in ghostsPositions]


        # feature vectors for capsules
        capsulesPositions = [cap[0] for cap in self.capsules]
        curDistancesC = [manhattanDistance(pacpos,capsulePosition) for capsulePosition in capsulesPositions]
        distanceCapsules = [manhattanDistance(newPosition,capsulePosition) for capsulePosition in capsulesPositions]

        distances = np.array(distanceGhosts+distanceCapsules)
        curDistances = np.array(curDistancesG+curDistancesC)

        candiAns = (distan