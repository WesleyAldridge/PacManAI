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
        - getQValue
        - computeValueFromQValues
        - computeActionFromQValues
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
        # util.raiseNotDefined()

        self.Q = util.Counter()

        # "You don't know the transitions T(s,a,s')
        # You don't know the rewards R(s,a,s')
        # Goal: learn the state values
        # Learner is "along for the ride"
        # No choice about what actions to take
        # Just execute the policy and learn from experience
        # This is NOT offline planning! You actually take actions in the world."
        # slide 19, "10_Reinforcement Learning 1.pptx"

        # sample1 = R(s,pi(s),s'1) + gamma*V(s'1)
        # sample2 = R(s,pi(s),s'2) + gamma*V(s'2)
        # samplen = R(s,pi(s),s'n) + gamma*V(s'n)
        # Vk+1(s) = 1/n * sum(samples)
        # Update V(s) each time we experience a transition (s, a, s', r)
        # Likely outcomes s' will contribute updates more often


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        legalActions = self.getLegalActions(state)

        if not legalActions:
            return 0.0

        maxVal = float("-inf")
        for action in legalActions:
            currentValue = self.getQValue(state,action)
            if currentValue > maxVal:
                maxVal = currentValue
        return maxVal


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # "Note: For computeActionFromQValues, you should break ties
        #  randomly for better behavior. The random.choice() function
        #  will help."

        legalActions = self.getLegalActions(state)

        if not legalActions:
            return None

        bestAction = None
        maxVal = float("-inf")
        for action in legalActions:
            currentValue = self.getQValue(state,action)
            if currentValue > maxVal:
                maxVal = currentValue
                bestAction = action
        return bestAction


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
        #util.raiseNotDefined()

        maxVal = float("-inf")
        bestAction = None

        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            # take the best policy action
            for action in legalActions:
                qValue = self.getQValue(state, action)
                if qValue > maxVal:
                    maxVal = qValue
                    bestAction = action
            return bestAction


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # slide 25 of "10_Reinforcement Learning 1.pptx":
        # sample of V(s):  sample = R(s,pi(s),s') + gamma * V(s')
        # update to V(s):  V(s) = (1-alpha)V(s) + alpha(sample)
        # same update   :  V(s) = V(s) + alpha(sample - V(s))
        # alpha is the rate of learning
        currentValue = self.getQValue(state,action)
        sample = reward + self.discount * self.getValue(nextState)
        self.Q[(state, action)] = currentValue + self.alpha * (sample - currentValue)

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
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        features = self.featExtractor.getFeatures(state, action)

        self.Q[(state, action)] = 0
        for feature in features:
            self.Q[(state, action)] += features[feature] * self.weights[feature]

        return self.Q[(state, action)]

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # pseudocode given in slide 23 of "11_Reinforcement Learning II.pptx" on webcourses
        # but also in the Q8 description. The formulas are given there as well.

        currentValue = self.getQValue(state, action)
        difference = (reward + self.discount * self.getValue(nextState)) - currentValue
        features = self.featExtractor.getFeatures(state, action)

        for feature in features:
            currentWeight = self.weights[feature]
            self.weights[feature] = currentWeight + (self.alpha*difference*features[feature])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
