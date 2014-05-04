from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import pickle
from sklearn.externals import joblib
import sys

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


class KDA9000Agent(BaseStudentAgent):
    '''
    Our actual agent
    '''
    ghostsdict = None
    class_to_value = [29.9735,49.3226,150.649,19.9242]

    g = lambda self,x,y: np.sign(x)/pow((pow(x,2)+pow(y,2)+1.),0.5)#/(np.absolute(x)+1.)
    f = [lambda self,s : 0 for i in range(24)]
    f[0] = lambda self,s : 0 #s.getPacmanState().getPosition()[0]
    f[1] = lambda self,s : 0 #s.getPacmanState().getPosition()[1]
    f[2] = lambda self,s : self.g((self.ghostsdict['bad'][0].getPosition()[0] - s.getPacmanState().getPosition()[0]), (self.ghostsdict['bad'][0].getPosition()[1] - s.getPacmanState().getPosition()[1]))
    f[3] = lambda self,s : self.g((self.ghostsdict['bad'][0].getPosition()[1] - s.getPacmanState().getPosition()[1]), (self.ghostsdict['bad'][0].getPosition()[0] - s.getPacmanState().getPosition()[0]))
    f[4] = lambda self,s : self.g((self.ghostsdict['good'][0][0].getPosition()[0] - s.getPacmanState().getPosition()[0]), (self.ghostsdict['good'][0][0].getPosition()[1] - s.getPacmanState().getPosition()[1]))
    f[5] = lambda self,s : self.g((self.ghostsdict['good'][0][0].getPosition()[1] - s.getPacmanState().getPosition()[1]), (self.ghostsdict['good'][0][0].getPosition()[0] - s.getPacmanState().getPosition()[0]))
    f[6] = lambda self,s : self.g((self.ghostsdict['good'][1][0].getPosition()[0] - s.getPacmanState().getPosition()[0]), (self.ghostsdict['good'][1][0].getPosition()[1] - s.getPacmanState().getPosition()[1]))
    f[7] = lambda self,s : self.g((self.ghostsdict['good'][1][0].getPosition()[1] - s.getPacmanState().getPosition()[1]), (self.ghostsdict['good'][1][0].getPosition()[0] - s.getPacmanState().getPosition()[0]))
    f[8] = lambda self,s : self.g((self.ghostsdict['good'][2][0].getPosition()[0] - s.getPacmanState().getPosition()[0]), (self.ghostsdict['good'][2][0].getPosition()[1] - s.getPacmanState().getPosition()[1]))
    f[9] = lambda self,s : self.g((self.ghostsdict['good'][2][0].getPosition()[1] - s.getPacmanState().getPosition()[1]), (self.ghostsdict['good'][2][0].getPosition()[0] - s.getPacmanState().getPosition()[0]))
    f[10] = lambda self,s : 0
    f[11] = lambda self,s : 0
    f[12] = lambda self,s : 0
    f[13] = lambda self,s : 0
    f[14] = lambda self,s : 0
    f[15] = lambda self,s : 0
    f[16] = lambda self,s : 0
    f[17] = lambda self,s : 0
    f[18] = lambda self,s : 0
    f[19] = lambda self,s : 0
    f[20] = lambda self,s : 0
    f[21] = lambda self,s : 0
    f[22] = lambda self,s : self.f[2](self,s) if s.scaredGhostPresent() else 0
    f[23] = lambda self,s : self.f[3](self,s) if s.scaredGhostPresent() else 0

    J = len(f)*5
    # thetas = np.zeros(J)
    fN_init = [0,0]+[0,-1,0,1,0,1,0,1]+([0]*12)+[0,1]
    fE_init = [0,0]+[-1,0,1,0,1,0,1,0]+([0]*12)+[1,0]
    fS_init = [0,0]+[0,1,0,-1,0,-1,0,-1]+([0]*12)+[0,-1]
    fW_init = [0,0]+[1,0,-1,0,-1,0,-1,0]+([0]*12)+[-1,0]
    fSt_init = [0]*24
    thetas = np.array(fN_init+fE_init+fS_init+fW_init+fSt_init,dtype=np.float_)
    prev_state = None
    prev_action = None
    optimal_action = None
    prev_score = 0
    clfGhost = None
    alpha = 0.001
    gamma = 1-pow(10,-10)
    dirs = [Directions.NORTH,Directions.EAST,Directions.SOUTH,Directions.WEST,Directions.STOP]
    t = 1
    dt = 0.1

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

    def classifyGhost(self, feat_v, quad):
        clf_v = np.insert(feat_v,0,quad)
        return int(self.clfGhost.predict(clf_v)[0])

    def classifyCapsule(self, clf, feat_v):
        pass

    # returns numpy array of [f1(s,a), f2(s,a), ..., fj(s,a)]
    def get_regression_feature(self, observedState, action):
        reg_features = np.zeros(self.J)
        j = self.J/5
        if action == Directions.NORTH:
            reg_features[0:j] = map(lambda feat:feat(self,observedState),self.f)
        elif action == Directions.EAST:
            reg_features[j:2*j] = map(lambda feat:feat(self,observedState),self.f)
        elif action == Directions.SOUTH:
            reg_features[2*j:3*j] = map(lambda feat:feat(self,observedState),self.f)
        elif action == Directions.WEST:
            reg_features[3*j:4*j] = map(lambda feat:feat(self,observedState),self.f)
        else:
            assert(action == Directions.STOP)
            reg_features[4*j:5*j] = map(lambda feat:feat(self,observedState),self.f)
        return reg_features

    # returns Q(state, action)
    def Q_sa(self, state, action):
        return np.dot(self.thetas,self.get_regression_feature(state,action))

    def get_target(self, curr_state, prev_state, prev_action):
        reward = curr_state.getScore() - prev_state.getScore()
        Qsas = map(lambda a: self.Q_sa(curr_state, a), self.dirs)
        print Qsas
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
                    continue
                else:
                    is_new = True
                    for gs_quad in self.ghostsdict['good']:
                        if (new_gs.getFeatures() == gs_quad[0].getFeatures()).all():
                            new_good_ghosts.append(gs_quad)
                            is_new = False
                            break
                    if is_new:
                        clas = self.classifyGhost(new_gs.getFeatures(), observedState.getGhostQuadrant(new_gs))
                        if clas == 5:
                            self.ghostsdict['bad'] = (new_gs, observedState.getGhostQuadrant(new_gs), -1000.)
                        else:
                            new_good_ghosts.append((new_gs, observedState.getGhostQuadrant(new_gs), self.class_to_value[clas]))
                        print "NEW GHOST!!!"

            # sort good ghosts in decreasing order
            if (len(new_good_ghosts) < 3):
                for new_gs in new_ghost_states:
                    print new_gs
                sys.exit()
            self.ghostsdict['good'] = sorted(new_good_ghosts,key=lambda x:-x[2])
        
        # return immediately if in None case (beginning of game)
        if self.prev_state == None or self.prev_action == None:
            assert(self.prev_state == self.prev_action)
            self.prev_state = observedState
            self.prev_action = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
            return self.prev_action

        feat_v = self.get_regression_feature(self.prev_state, self.prev_action)
        target_sa = self.get_target(observedState, self.prev_state, self.prev_action)

        # index given by [north, east, south, west, stop]
        index_factor = self.dirs.index(self.prev_action)
        # update only the thetas that are nonzeroes determined by prev_action, and the indices are edetermined by
        # index_factor
        for j in xrange(index_factor*self.J/5,(index_factor+1)*self.J/5):
            self.thetas[j] = self.thetas[j] + self.alpha*(target_sa - self.Q_sa(self.prev_state, self.prev_action)) \
                *self.f[j % (self.J/5)](self,self.prev_state)
        
        self.prev_state = observedState

        epsilon = 1/self.t
        if np.random.rand() < 1. - epsilon:
            self.prev_action = self.optimal_action
        else:
            self.prev_action = observedState.getLegalPacmanActions()[np.random.randint(len(observedState.getLegalPacmanActions()))]
        self.prev_score = observedState.getScore()
        self.t += self.dt
        print self.thetas
        print 1. - epsilon
        #print observedState.getPacmanPosition()
        return self.prev_action

