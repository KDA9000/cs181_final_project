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

    def best_move(self, legal_moves, curr_pos, dest):
        best_action = Direction.STOP
        best_dist = np.inf
        lst = []
        for move in legal_moves:
            if move == Direcitions.STOP:
                continue
            successor_pos = Actions.getSuccessor(curr_pos, move)
            distance = manhattanDistance(successor_pos, dest)
            if distance < best_dist:
                lst = [move]
                best_dist = distance
            elif distance == best_dist:
                lst.append(move)
        if lst == []:
            return Distance.STOP
        else:
            lst[np.random.randint(len(lst))]

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
            return self.best_move(legal_actions, pacmanPosition, bad_ghost_pos)
        # go towards good capsule unless move will immediately endanger ghost
        else:
            legal_actions = observedState.getLegalPacmanActions()
            good_caps = self.get_good_caps(observedState)
            dest = good_caps[0][0]
            # Stop if next to good capsule
            if manhattanDistance(pacmanPosition, dest) == 1:
                if manhattanDistance(pacmanPosition, bad_ghost_pos) > 1:
                    return Directions.STOP
                else:
                    return self.best_move(legal_actions, pacmanPosition, dest)

            best_action = best_move(legal_actions, pacmanPosition, dest)            
            successor_pos = Actions.getSuccessor(pacmanPosition, best_action)

            if manhattanDistance(successor_pos, bad_ghost_pos) < 2:

                possible_actions = []
                for move in legal_actions:
                    if move == Directions.STOP:
                        continue
                    successor_pos = Actions.getSuccessor(pacmanPosition, move)
                    if manhattanDistance(successor_pos, bad_ghost_pos) > 1:
                        possible_actions.append(move)
                lst_len = len(possible_actions)
                if lst_len > 0:
                    best_action = possible_actions[np.random.randint(lst_len)]
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
    thetas = [12.072337981216242, -0.48824095638567144, -0.12732825706817857, -14.119137086538544] + [0.]*6
    prev_state = None
    prev_action = None
    optimal_action = None
    prev_score = 0
    clfGhost = None
    alpha = 0.00005
    gamma = 1-pow(10,-10)
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

        candiAns = (distances-curDistances)/np.absolute(curDistances.clip(1))
        if state.scaredGhostPresent():
            candiAns[0] = -candiAns[0]
        return candiAns

        # return np.concatenate((((distanceGhosts-curDistances)/np.square(curDistances.clip(1)),distanceGhosts)),axis=0)
    '''
    def mindist_ghost(self,state,action):
        pacState = state.getPacmanState()
        legalActions = state.getLegalPacmanActions()
        pacpos = pacState.getPosition()
        speed = 1.0

        actionVector = Actions.directionToVector( action , speed )
        ghostsPositions = [self.ghostsdict['bad'][0].getPosition()]
        for i in range(3):
            ghostsPositions.append(self.ghostsdict['good'][i][0].getPosition())
        newPosition = (pacpos[0]+actionVector[0], pacpos[1]+actionVector[1])
        return manhattanDistance(newPosition,ghostPosition)
    '''
    def __init__(self, *args, **kwargs):
        pass

    def registerInitialState(self, gameState):
        super(KDA9000Agent, self).registerInitialState(gameState)
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")
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


    # returns numpy array of [f1(s,a), f2(s,a), ..., fj(s,a)]
    def get_regression_feature(self, observedState, action):
        return self.ddist_ghost(observedState,action)

    # returns Q(state, action)
    def Q_sa(self, state, action):
        return np.dot(self.thetas,self.get_regression_feature(state,action))

    def get_target(self, curr_state, prev_state, prev_action):
        reward = curr_state.getScore() - prev_state.getScore() +5
        Qsas = map(lambda a: self.Q_sa(curr_state, a), self.dirs)
        tuples = zip(self.dirs,Qsas)
        tuples = sorted(tuples,key=lambda t:-t[1])
        valid_dirs = curr_state.getLegalPacmanActions()
        found_optimal = False
        maxQsa = None
        for tu in tuples:
            if tu[0] in valid_dirs:
                self.optimal_action = tu[0]
                found_optimal = True
                maxQsa = tu[1]
                break
        assert(found_optimal and maxQsa != None)
        return reward + self.gamma*maxQsa

    def chooseAction(self, observedState):
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


            # sort good ghosts in decreasing order
            if (len(new_good_ghosts) < 3):
                print "problem"
                for new_gs in new_ghost_states:
                    print new_gs
                print "end problem"
                sys.exit()
            self.ghostsdict['good'] = sorted(new_good_ghosts,key=lambda x:-x[2]+(x[1]+x[0].getFeatures())[1]/1000.)
        
        #update capsule information
        # caps = observedState.getCapsuleData()
        # pacPos = observedState.getPacmanState().getPosition()
        # relpos = [(cap[0][0]-pacPos[0],cap[0][1]-pacPos[1]) for cap in caps]
        # newcaps = [(caps[i][0],caps[i][1],relpos[i]) for i in range(len(caps))]
        # self.capsules = sorted(newcaps,key=lambda x:manhattanDistance(x[2],(0,0)))

        # sort the capsules first by potentially good capsules followed by bad 
        # capsules, and within good/bad capsules, sort by closest distance
        # to PacMan's current position first
        caps_and_class = [(cap, self.classifyCapsule(cap[1])) for cap in observedState.getCapsuleData()]
        good_caps = [cap_class[0] for cap_class in caps_and_class if cap_class[1]]
        bad_caps = [cap_class[0] for cap_class in caps_and_class if not cap_class[1]]
        sorted_good_caps = sorted(good_caps, key = lambda x: manhattanDistance(x[0], observedState.getPacmanPosition()))
        sorted_bad_caps = sorted(bad_caps, key = lambda x: manhattanDistance(x[0], observedState.getPacmanPosition()))
        sorted_all_caps = sorted_good_caps + sorted_bad_caps
        self.capsules = sorted_all_caps
        
        #self.capsules = sorted(observedState.getCapsuleData(),key=lambda x:x[1][1])

        # return immediately if in None case (beginning of game)
        if self.prev_state == None or self.prev_action == None:
            assert(self.prev_state == self.prev_action)
            self.prev_state = observedState
            self.prev_action = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.prev_action in observedState.getLegalPacmanActions())
            return self.prev_action

        trapped = self.trapped_dir
        if trapped != None:
            self.trapped_dir = None
            
            #assert(trapped in observedState.getLegalPacmanActions())
            #return trapped
            return observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]

        pacPos = observedState.getPacmanState().getPosition()
        wall = lambda x,y:observedState.hasWall(x,y)
        if wall(pacPos[0]+1,pacPos[1]) and wall(pacPos[0]-1,pacPos[1]) and wall(pacPos[0],pacPos[1]-1):
            self.trapped_dir = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.trapped_dir in observedState.getLegalPacmanActions())
            return self.trapped_dir
        elif wall(pacPos[0]+1,pacPos[1]) and wall(pacPos[0]-1,pacPos[1]) and wall(pacPos[0],pacPos[1]+1):
            self.trapped_dir = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.trapped_dir in observedState.getLegalPacmanActions())
            return self.trapped_dir
        elif wall(pacPos[0],pacPos[1]+1) and wall(pacPos[0],pacPos[1]-1) and wall(pacPos[0]+1,pacPos[1]):
            self.trapped_dir = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.trapped_dir in observedState.getLegalPacmanActions())
            return self.trapped_dir
        elif wall(pacPos[0],pacPos[1]+1) and wall(pacPos[0],pacPos[1]-1) and wall(pacPos[0]-1,pacPos[1]):
            self.trapped_dir = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.trapped_dir in observedState.getLegalPacmanActions())
            return self.trapped_dir     


        feat_v = self.get_regression_feature(self.prev_state, self.prev_action)
        target_sa = self.get_target(observedState, self.prev_state, self.prev_action)

        # index given by [north, east, south, west, stop]
        index_factor = self.dirs.index(self.prev_action)
        # update only the thetas that are nonzeroes determined by prev_action, and the indices are edetermined by
        # index_factor
        prevQ = self.Q_sa(self.prev_state, self.prev_action)
        for j in xrange(10):
            candi = self.thetas[j] + self.alpha*(target_sa - prevQ)*feat_v[j]
            if j==0 and candi >0:
                self.thetas[j] = candi
            elif candi <= 0:
                self.thetas[j] = candi

        # self.thetas[0] = self.thetas[0] + self.alpha*(target_sa - prevQ)*feat_v[0]
        
        self.prev_state = observedState

        epsilon = 1/self.t
        if np.random.rand() < 1. - epsilon:
            self.prev_action = self.optimal_action
            assert(self.prev_action in observedState.getLegalPacmanActions() or self.prev_action == None)
        else:
            self.prev_action = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            assert(self.prev_action in observedState.getLegalPacmanActions())
        self.prev_score = observedState.getScore()
        self.t += self.dt
        print self.thetas
        print self.t
        print 1. - epsilon
        #print observedState.getPacmanPosition()
        # print self.ghostsdict

        return self.prev_action



