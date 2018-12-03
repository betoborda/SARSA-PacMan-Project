# qlearningAgents.py
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

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
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
        #print("Q", Q)
        return Q
        #util.raiseNotDefined()

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
        #print("Probability ", self.epsilon)
        #print(util.flipCoin(self.epsilon))
        if legalActions:
            #take random action if true
            if util.flipCoin(self.epsilon):
                action=random.choice(legalActions)
            #take best policy
            else:
                action=self.getPolicy(state) 
        return action
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        #V(s)=Q(s,a)+alpha(R(s)+alpha*V(s')-Q(s,a))
        #print(self.getQValue(state, action))
        self.values[(state,action)]=self.getQValue(state, action)+self.alpha*(reward+self.discount*self.getValue(nextState)-self.getQValue(state,action))
        #print(self.values[(state,action)])
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

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
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
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
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #difference = (r + lambda * maxQ(s',a')) - Q(s, a)
        maxQ=self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * maxQ) - self.getQValue(state,action)

        #wi = wi + alpha* difference * fi(s, a)
        featureVectors=self.featExtractor.getFeatures(state,action)
        for eachFeature in featureVectors:
            f=featureVectors[eachFeature]
            self.weights[eachFeature] = self.weights[eachFeature] + self.alpha * difference * f
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass
