# valueIterationAgents.py
# -----------------------
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


import util

from CodeFiles.learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #for each iteration, go to each state and for each possible action in that state compute Q values
        #find the max Q value for each state  
        for iteration in range(self.iterations):
            currentValues= util.Counter()
            stateValues=self.mdp.getStates()
            for eachState in stateValues:
                maxQ=float("-inf")
                possibleActions=self.mdp.getPossibleActions(eachState)
                for eachAction in possibleActions:
                    newQ=self.computeQValueFromValues(eachState, eachAction)
                    if newQ>maxQ:
                        maxQ=newQ
                        currentValues[eachState]=maxQ
            self.values=currentValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q=0
        for eachState, eachProb in self.mdp.getTransitionStatesAndProbs(state, action):
            R=self.mdp.getReward(state, action, eachState)
            #print(eachState)
            #print(eachProb)
            #Q(s)=Q(s)+alpha*(R(s)+lambda*V(s'))
            Q=Q+eachProb*(R+self.discount*self.getValue(eachState))
        return Q   
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #return best action for the given state or None if there are no legal actions
        actions=self.mdp.getPossibleActions(state)
        #print(actions)
        bestAction=None
        maxQ=float("-inf")
        for eachAction in actions:
            Q=self.computeQValueFromValues(state,eachAction)
            if Q>maxQ:
                maxQ=Q
                bestAction=eachAction
        return bestAction 
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
