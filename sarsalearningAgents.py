# sarsalearningAgents.py
# ------------------

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class SarsaLearningAgent(ReinforcementAgent):
    print("In Sarsa Learning Agent")
    """
      Sarsa-Learning Agent

      Functions you should fill in:
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        Q = 0.0
        currentStateAction = (state,action)
        if currentStateAction in self.values:
           Q=self.values[currentStateAction]
        return Q

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        #return max q value for the given state or 0.0 if there are no legal actions
        actions=self.getLegalActions(state)
        #print(actions)
        if actions:
            maxQ=float("-inf")
        else:
            maxQ=0.0
        for eachAction in actions:
            Q=self.getQValue(state, eachAction)
            if Q>maxQ:
                maxQ=Q
        return maxQ 

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        #return max action for the given state or None if there are no legal actions
        actions=self.getLegalActions(state)
        bestAction=None
        maxQ=float("-inf")
        bestAction=random.choice(actions)
        for eachAction in actions:
            Q=self.getQValue(state, eachAction)
            if Q>maxQ:
                maxQ=Q
                bestAction=eachAction
        return bestAction 


    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions:
            action=self.getPolicy(state) 
        return action

    def update(self, state, action, nextState, reward):
        action2=self.getAction(nextState)
        #V(s)=Q(s,a)+alpha(R(s)+alpha*V(s')-Q(s,a))
        if(action2!='none'):
            Q=self.getQValue(nextState, action2)
        else:
            Q=0.0
        self.values[(state,action)]=self.getQValue(state, action)+self.alpha*(reward+self.discount*Q-self.getQValue(state,action))
        #print(self.values[(state,action)])

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    
    def getValue(self, state):
        return self.computeValueFromQValues(state)



class PacmanSarsaAgent(SarsaLearningAgent):

    def __init__(self,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p SarsaQLearningAgent 

        alpha    - learning rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of SarsaLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = SarsaLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateSarsaAgent(PacmanSarsaAgent):
    """
       ApproximateSarsaLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()


    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        #Q(s,a) = Summation from i=1 to n fi(s, a)*wi
        Q=0
        featureVectors=self.featExtractor.getFeatures(state,action)
        for eachFeature in featureVectors:
            #print("Feature ",eachFeature)
            #print("Weights ",self.weights[eachFeature])
            f=featureVectors[eachFeature]
            w=self.weights[eachFeature]
            Q=Q+(f * w)
        return Q

    def update(self, state, action, nextState, reward):
        action2=self.getAction(nextState)

        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #difference = (r + lambda * (s',a')) - Q(s, a)
        if(action2!='none'):
            Q=self.getQValue(nextState, action2)
        else:
            Q=0.0
        difference = (reward + self.discount * Q) - self.getQValue(state,action)

        #wi = wi + alpha* difference * fi(s, a)
        featureVectors=self.featExtractor.getFeatures(state,action)
        for eachFeature in featureVectors:
            f=featureVectors[eachFeature]
            self.weights[eachFeature] = self.weights[eachFeature] + self.alpha * difference * f
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanSarsaAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass
