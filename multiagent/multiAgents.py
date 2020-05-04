# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #newLegalMoves = []
        #for i in legalMoves:
        #    newLegalMoves.append(i)
        #if len(newLegalMoves) > 1:
        #    newLegalMoves = newLegalMoves.remove('Stop')
        #legalMovesWithoutStop = legalMoves.remove('Stop')
        #print("legalMoves are ")
        #print(legalMoves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #chosenIndex = bestIndices[0]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        firstFoodPos = (0,0)
        rowIndex = 0
        colIndex = 0
        #foodPositions =[]
        for row in newFood:
            rowIndex += 1
            for col in row:
                colIndex += 1
                if col == True:
                    coord = (rowIndex, colIndex)
                    #foodPositions.append((rowIndex, colIndex))
                    firstFoodPos = (rowIndex, colIndex)
                    break
            else:
                colIndex = 0
                continue
            break

        distanceFromFirstFood = abs(newPos[0] - firstFoodPos[0]) + abs(newPos[1] - firstFoodPos[1])
        distanceFromFirstFood = distanceFromFirstFood / 2
        #foodDistances = [abs(newPos[0] - food[0]) + abs(newPos[1] - food[1]) for food in foodPositions]
        #print(foodDistances)

        # Manhattan distance:
        ghostDistances = [abs(newPos[0] - ghost[0]) + abs(newPos[1] - ghost[1]) for ghost in newGhostPositions]

        #if min(newScaredTimes) > 0:
            #score = successorGameState.getScore() + (min(ghostDistances) + (1/distanceFromFirstFood)
        if min(ghostDistances) > 3:
            if distanceFromFirstFood == 0:
                score = successorGameState.getScore() + 2.0
            else:
                score = successorGameState.getScore() + (1.0/distanceFromFirstFood)
        else:
            if min(ghostDistances) == 0:
                score = successorGameState.getScore() - 200
            else:
                score = successorGameState.getScore() - 100*(1.0 / min(ghostDistances))

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        currentState = gameState
        startingDepth = 0

        # max_value is for Pacman, min_value is for ghosts
        # pseudocode for functions is on slides 16 and 17 of "6_Adversarial Search.pptx" on Webcourses
        def max_value(currentState, currentDepth):
            # "if the state is a terminal state: return the state's utility"
            if currentState.isWin() or currentState.isLose():
                return currentState.getScore()

            # "initialize v = -infinity"
            v = float("-inf")

            minimax = 'Stop'

            legalMoves = currentState.getLegalActions(0)
            # "for each successor of state:"
            for action in legalMoves:
                minVal = min_value(currentState.generateSuccessor(0, action), currentDepth, 1)
                if minVal > v:
                    # "v = max(v, value(successor))"
                    v = minVal
                    minimax = action

            if currentDepth == 0:
                # we have returned from the recursive calls
                return minimax
            else:
                # still recursing
               return v


        def min_value(currentState, currentDepth, currentAgent):
            # "if the state is a terminal state: return the state's utility"
            if currentState.isLose() or currentState.isWin():
                return currentState.getScore()

            nextAgent = currentAgent + 1
            if currentAgent == currentState.getNumAgents() - 1:
                # This is the last agent, next we will start over with Pacman:
                nextAgent = 0

            # "initialize v = +infinity"
            v = float("inf")
            maxVal = v

            legalMoves = currentState.getLegalActions(currentAgent)
            for action in legalMoves:

                if nextAgent == 0:
                    # No more ghosts, next agent is Pacman.
                    if currentDepth == self.depth - 1:
                        # There are no more ghosts, AND we have reached max depth, so we're finished!
                        maxVal = self.evaluationFunction(currentState.generateSuccessor(currentAgent, action))
                    else:
                        # next agent is Pacman, but we're not at max depth yet, so go deeper:
                        maxVal = max_value(currentState.generateSuccessor(currentAgent, action), currentDepth + 1)
                else:
                    # still more ghosts, do min_value on next ghost:
                    maxVal = min_value(currentState.generateSuccessor(currentAgent, action), currentDepth, nextAgent)

                # "v = min(v, value(successor))"
                v = min(v, maxVal)
            return v

        return max_value(currentState, startingDepth)


# pseudocode for alpha-beta pruning is on slide 40 of "6_Adversarial Search.pptx" on Webcourses
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        currentState = gameState
        startingDepth = 0
        a = float("-inf")
        B = float("inf")

        # alpha (a): max's best option
        # beta  (B): min's best option

        # max_value is for Pacman, min_value is for ghosts
        # pseudo code for functions is on slides 16 and 17 of "6_Adversarial Search.pptx" on Webcourses
        def max_value(currentState, currentDepth, a, B):
            # "if the state is a terminal state: return the state's utility:"
            if currentState.isWin() or currentState.isLose():
                return currentState.getScore()

            # "initialize v = -infinity"
            v = float("-inf")

            minimax = 'Stop'

            legalMoves = currentState.getLegalActions(0)
            # "for each successor of state:"
            for action in legalMoves:
                minVal = min_value(currentState.generateSuccessor(0, action), currentDepth, 1, a, B)
                if minVal > v:
                    # "v = max(v, value(successor))"
                    v = minVal
                    minimax = action

                # "You must NOT prune on equality in order to match the set of states explored by our autograder."
                # "The pseudo-code below represents the algorithm you should implement for this question."
                # "if v > B return v"
                # - from instructions
                if v > B:
                    return v

                # "a = max(a, v)"
                a = max(a, v)

            if currentDepth == 0:
                # we have returned from the recursive calls
                return minimax
            else:
                # still recursing
                return v


        def min_value(currentState, currentDepth, currentAgent, a, B):
            # if the state is a terminal state: return the state's utility
            if currentState.isLose() or currentState.isWin():
                return currentState.getScore()

            nextAgent = currentAgent + 1
            if currentAgent == currentState.getNumAgents() - 1:
                # This is the last agent, next we will start over with Pacman:
                nextAgent = 0

            # "initialize v = infinity"
            v = float("inf")
            maxVal = v

            legalMoves = currentState.getLegalActions(currentAgent)
            # "for each successor of state:"
            for action in legalMoves:

                if nextAgent == 0:
                    # No more ghosts, next agent is Pacman.
                    if currentDepth == self.depth - 1:
                        # There are no more ghosts, AND we have reached max depth, so we're finished!
                        maxVal = self.evaluationFunction(currentState.generateSuccessor(currentAgent, action))
                    else:
                        # next agent is Pacman, but we're not at max depth yet, so go deeper:
                        maxVal = max_value(currentState.generateSuccessor(currentAgent, action), currentDepth + 1, a, B)
                else:
                    # still more ghosts, continue with next ghost:
                    maxVal = min_value(currentState.generateSuccessor(currentAgent, action), currentDepth, nextAgent, a, B)

                # "v = min(v, value(successor))"
                v = min(v, maxVal)

                # "You must NOT prune on equality in order to match the set of states explored by our autograder."
                # "The pseudo-code below represents the algorithm you should implement for this question."
                # "if v < a return v"
                # - from instructions
                if v < a:
                    return v

                # "B = min (B, v)"
                B = min(B, v)
            return v

        return max_value(currentState, startingDepth, a, B)


# pseudo code for expectimax is on slide 7 of "7_Expectimax Search and Utilities.pptx" on Webcourses
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        currentState = gameState
        startingDepth = 0

        # max_value is for Pacman, exp_value is for ghosts
        def max_value(currentState, currentDepth):
            # "if the state is a terminal state: return the state's utility"
            if currentState.isWin() or currentState.isLose():
                return currentState.getScore()

            # "initialize v = -infinity"
            v = float("-inf")

            minimax = 'Stop'

            legalMoves = currentState.getLegalActions(0)
            # "for each successor of state:"
            for action in legalMoves:
                expVal = exp_value(currentState.generateSuccessor(0, action), currentDepth, 1)
                if expVal > v:
                    # "v = max(v, value(successor))"
                    v = expVal
                    minimax = action

            if currentDepth == 0:
                # we have returned from the recursive calls
                return minimax
            else:
                # still recursing
                return v

        def exp_value(currentState, currentDepth, currentAgent):
            # if the state is a terminal state: return the state's utility
            if currentState.isLose() or currentState.isWin():
                return currentState.getScore()

            nextAgent = currentAgent + 1
            if currentAgent == currentState.getNumAgents() - 1:
                # No more ghosts, next agent is Pacman.
                nextAgent = 0

            # "initialize v = 0"
            v = 0

            legalMoves = currentState.getLegalActions(currentAgent)
            # "for each successor of state:"
            for action in legalMoves:
                # "p = probability(successor)"
                # (assuming same probability for each successor)
                probability = 1.0/len(legalMoves)
                if nextAgent == 0:
                    # No more ghosts, next agent is Pacman.
                    if currentDepth == self.depth - 1:
                        # There are no more ghosts, AND we have reached max depth, so we're finished!
                        v = self.evaluationFunction(currentState.generateSuccessor(currentAgent, action))
                        v += probability * v
                    else:
                        # next agent is Pacman, but we're not at max depth yet, so go deeper:
                        v = max_value(currentState.generateSuccessor(currentAgent, action), currentDepth + 1)
                        v += probability * v
                else:
                    # still more ghosts, continue with next ghost
                    v = exp_value(currentState.generateSuccessor(currentAgent, action), currentDepth, nextAgent)
                    v += probability * v
            return v
        return max_value(currentState, startingDepth)



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = currentGameState.getFood().asList()
    ghostPositions = currentGameState.getGhostPositions()

    foodPositions = []
    foodDistances = []
    currentScore = currentGameState.getScore()

    if currentGameState.isWin():
        return 10000
    if currentGameState.isLose():
        return -10000

    for food in foodList:
        foodPositions.append(food)
    for position in foodPositions:
        foodDistances.append(abs(pos[0] - position[0]) + abs(pos[1] - position[1]))
    if len(foodList) > 0:
        nearestFood = min(foodDistances)
        whereMostOfTheFoodIs = sum(foodDistances)
        foodLeft = len(foodList)
    else:
        nearestFood = 0
        whereMostOfTheFoodIs = 0
        foodLeft = 0

    ghostDistances = [abs(pos[0] - ghost[0]) + abs(pos[1] - ghost[1]) for ghost in ghostPositions]

    if min(ghostDistances) > 3:
        score = currentGameState.getScore() + (1.0 / (10.0 * foodLeft + nearestFood + whereMostOfTheFoodIs))
        #score = currentGameState.getScore() - (0.30 *(10*foodLeft + nearestFood + whereMostOfTheFoodIs))
        if foodLeft == 0 or currentGameState.isWin():
            score = 1000000
    else:
        score = currentGameState.getScore() - 10*(1.0/min(ghostDistances)) - foodLeft

    return score

# Abbreviation
better = betterEvaluationFunction

