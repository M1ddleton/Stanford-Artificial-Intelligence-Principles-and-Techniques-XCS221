from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        # BEGIN_HIDE
        # END_HIDE

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # BEGIN_HIDE
        # END_HIDE
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
        pass

        # ### START CODE HERE ###

        def maxVal(gameState, agentIndex, currentDepth):
            v = (float('-inf'), Directions.STOP)
            next_agent = (agentIndex + 1)
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth)
                if newVal > v[0]:
                    v = (newVal, action)
            return v

        def minVal(gameState, agentIndex, currentDepth):
            v = (float('inf'), Directions.STOP)
            next_agent = (agentIndex + 1)
            if next_agent == gameState.getNumAgents():
                next_agent = 0
                currentDepth -= 1
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth)
                if newVal < v[0]:
                    v = (newVal, action)
            return v

        def minimaxValue(gameState, agentIndex, currentDepth):
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            if currentDepth <= 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxVal(gameState, agentIndex, currentDepth)[0]
            else:
                return minVal(gameState, agentIndex, currentDepth)[0]
        return maxVal(gameState, 0, self.depth)[1]

        # ### END CODE HERE ###


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        pass
        # ### START CODE HERE ###
        def maxVal(gameState, agentIndex, currentDepth,alpha, beta):
            v = (float('-inf'), Directions.STOP)
            next_agent = (agentIndex + 1)
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth, alpha, beta)
                if newVal > v[0]:
                    v = (newVal, action)
                alpha = max(alpha, v[0])
                if v[0] > beta:
                    return v
            return v

        def minVal(gameState, agentIndex, currentDepth, alpha, beta):
            v = (float('inf'), Directions.STOP)
            next_agent = (agentIndex + 1)
            if next_agent == gameState.getNumAgents():
                next_agent = 0
                currentDepth -= 1
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth, alpha, beta)
                if newVal < v[0]:
                    v = (newVal, action)
                beta = min(beta, v[0])
                if v[0] < alpha:
                    return v
            return v

        def minimaxValue(gameState, agentIndex, currentDepth, alpha, beta):
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            if currentDepth <= 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxVal(gameState, agentIndex, currentDepth, alpha, beta)[0]
            else:
                return minVal(gameState, agentIndex, currentDepth, alpha, beta)[0]
        return maxVal(gameState, 0, self.depth, float('-inf'), float('inf'))[1]


        # ### END CODE HERE ###


######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 3)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        pass
        # ### START CODE HERE ###
        def maxVal(gameState, agentIndex, currentDepth):
            v = (float('-inf'), Directions.STOP)
            next_agent = (agentIndex + 1)
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth)
                if newVal > v[0]:
                    v = (newVal, action)
            return v

        def expectimaxValue(gameState, agentIndex, currentDepth):
            v = [0.0]
            next_agent = (agentIndex + 1)
            if next_agent == gameState.getNumAgents():
                next_agent = 0
                currentDepth -= 1
            for action in gameState.getLegalActions(agentIndex):
                next_gameState = gameState.generateSuccessor(agentIndex, action)
                newVal = minimaxValue(next_gameState, next_agent, currentDepth)
                v.append(newVal)
            v.pop(0)
            return sum(v)/len(v)

        def minimaxValue(gameState, agentIndex, currentDepth):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            if currentDepth <= 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxVal(gameState,agentIndex,currentDepth)[0]
            else:
                return expectimaxValue(gameState, agentIndex, currentDepth)
        return maxVal(gameState, 0, self.depth)[1]




        # ### END CODE HERE ###


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):

    """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """
    pass
    # ### START CODE HERE ###
    pacmanPos = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    hunting_ghosts = []
    scared_ghosts = []
    scared_time = []
    for ghost in ghost_states:
        if ghost.scaredTimer:
            scared_time.append(ghost.scaredTimer)
            scared_ghosts.append(ghost)
        else:
            hunting_ghosts.append(ghost)
    food = currentGameState.getFood()
    food_list = food.asList()
    capsules = currentGameState.getCapsules()
    remain_food = len(food_list)
    remain_capsules = len(capsules)
    currentScore = currentGameState.getScore()
##############################################################
    closest_food = float("inf")
    invDistanceToClosestFood = 0
    for item in food_list:
        dist = util.manhattanDistance(pacmanPos, item)
        if dist < closest_food:
            closest_food = dist
    if closest_food > 0:
        inv_dist_to_closest_food = 1 / closest_food
    if len(food_list) < 3:
        inv_dist_to_closest_food = 100000
    if len(food_list) == 1:
        inv_dist_to_closest_food = 500000

#################################################################
    closest_capsules = float("inf")
    dist_to_closest_capsules = 0
    if remain_capsules == 0:
        closest_capsules = 0
    for item in capsules:
        dist = util.manhattanDistance(pacmanPos, item)
        if dist < closest_capsules:
            closest_capsules = dist
    if closest_capsules > 0:
        dist_to_closest_capsules = 1 / closest_capsules

##################################################################
    dist_to_hunting_ghost = float("inf")
    for ghost in hunting_ghosts:
        dist = util.manhattanDistance(pacmanPos, ghost.getPosition())
        if dist < dist_to_hunting_ghost:
            dist_to_hunting_ghost = dist
    inv_hunting_ghost = 0
    if dist_to_hunting_ghost > 0:
        inv_hunting_ghost = 1.0 / dist_to_hunting_ghost

    if len(scared_ghosts) == 0:
        dist_to_scared_ghost = 0
        scaredTime = 0
    else:
        dist_to_scared_ghost = float("inf")
        for ghost in scared_ghosts:
            dist = util.manhattanDistance(pacmanPos, ghost.getPosition())
            if dist < dist_to_scared_ghost:
                dist_to_scared_ghost = dist
        scaredTime = scared_time[0]
    inv_scared_ghost = 0
    if dist_to_scared_ghost > 0:
        inv_scared_ghost = 1.0 / dist_to_scared_ghost

######################################################################
    final_score = currentGameState.getScore() \
            - 7 * inv_hunting_ghost \
            + 15 * scaredTime * inv_scared_ghost \
            - 3 * remain_food \
            + 5 * dist_to_closest_capsules \
            - 1 * closest_food
    return final_score

    # ### END CODE HERE ###


# Abbreviation
better = betterEvaluationFunction
