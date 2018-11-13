# sarsalearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        - self.epsilon (exploration prob)
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
        "*** YOUR CODE HERE ***"
        Q = 0.0
        currentStateAction = (state,action)
        if currentStateAction in self.values:
           Q=self.values[currentStateAction]
        return Q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #return max action for the given state or None if there are no legal actions
        actions=self.getLegalActions(state)
        #print(actions)
        bestAction=None
        maxQ=float("-inf")
        for eachAction in actions:
            Q=self.getQValue(state, eachAction)
            if Q>maxQ:
                maxQ=Q
                bestAction=eachAction
        return bestAction 
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions:
            #take random action if true
            if util.flipCoin(self.epsilon):
                action=random.choice(legalActions)
                #decaying epsilon
                self.epsilon = 0.1*self.epsilon
            #else take best policy
            else:
                action=self.getPolicy(state) 
        return action
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        action2=self.getAction(nextState)
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #V(s)=Q(s,a)+alpha(R(s)+alpha*V(s')-Q(s,a))
        #print(self.getQValue(state, action))
        #Q=self.getQValue(nextState, action2)
        Q=self.getQValue(nextState, action2)
        self.values[(state,action)]=self.getQValue(state, action)+self.alpha*(reward+self.discount*Q-self.getQValue(state,action))
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)


class SarsaQAgent(SarsaLearningAgent):

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p SarsaQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
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


class ApproximateSarsaAgent(SarsaQAgent):
    """
       ApproximateSarsaLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        SarsaQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #Q(s,a) = Summation from i=1 to n fi(s, a)*wi
        Q=0
        featureVectors=self.featExtractor.getFeatures(state,action)
        for eachFeature in featureVectors:
            f=featureVectors[eachFeature]
            w=self.weights[eachFeature]
            Q=Q+(f * w)
        return Q
        #util.raiseNotDefined()

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
        SarsaQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
