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
import sys

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
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        eval = 0
        
        closestGhostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        foods = currentGameState.getFood().asList()
        if foods:
          closestFoodDist = min([manhattanDistance(newPos, food) for food in foods])
        """
        capsules = currentGameState.getCapsules()
        if capsules:
          closestCapsuleDist = min([manhattanDistance(newPos, capsule) for capsule in capsules])
        """
        if closestGhostDist > 3 or newScaredTimes[0] > 0:
          eval += 20 - closestFoodDist
        else:
          eval += closestGhostDist
        
        if action == "Stop":
          eval -= 1

        return eval

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
        return self.getMaxValue(gameState, 0, 0)[1]

    def getValue(self, state, agent, depth):
      if depth == self.depth:
        return self.evaluationFunction(state), None
      elif agent == 0:
        return self.getMaxValue(state, agent, depth)
      else:
        return self.getMinValue(state, agent, depth)

    def getMaxValue(self, state, agent, depth):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

      if len(successorStates) == 0:
        return self.evaluationFunction(state), None
      else:
        finalValue = -sys.maxint, None
        for state, action in successorStates:
          value = self.getValue(state, nextAgent, nextDepth)
          if value[0] > finalValue[0]:
            finalValue = value[0], action
        return finalValue

    def getMinValue(self, state, agent, depth):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

      if len(successorStates) == 0:
        return self.evaluationFunction(state), None
      else:
        finalValue = sys.maxint, None
        for state, action in successorStates:
          value = self.getValue(state, nextAgent, nextDepth)
          if value[0] < finalValue[0]:
            finalValue = value[0], action
        return finalValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.getMaxValue(gameState, 0, 0, -sys.maxint, sys.maxint)[1]

    def getValue(self, state, agent, depth, alpha, beta):
      if depth == self.depth:
        return self.evaluationFunction(state), None
      elif agent == 0:
        return self.getMaxValue(state, agent, depth, alpha, beta)
      else:
        return self.getMinValue(state, agent, depth, alpha, beta)

    def getMaxValue(self, state, agent, depth, alpha, beta):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      #successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
      actions = state.getLegalActions(agent)
      
      if len(actions) == 0:
        return self.evaluationFunction(state), None
      else:
        finalValue = -sys.maxint, None
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = self.getValue(successor, nextAgent, nextDepth, alpha, beta)
          if value[0] > finalValue[0]:
            finalValue = value[0], action
          if finalValue[0] > beta:
            return finalValue
          alpha = max(alpha, finalValue[0])
        return finalValue

    def getMinValue(self, state, agent, depth, alpha, beta):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      #successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
      actions = state.getLegalActions(agent)
      
      if len(actions) == 0:
        return self.evaluationFunction(state), None
      else:
        finalValue = sys.maxint, None
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = self.getValue(successor, nextAgent, nextDepth, alpha, beta)
          if value[0] < finalValue[0]:
            finalValue = value[0], action
          if finalValue[0] < alpha:
            return finalValue
          beta = min(beta, finalValue[0])
        return finalValue

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
        return self.getMaxValue(gameState, 0, 0)[1]

    def getValue(self, state, agent, depth):
      if depth == self.depth:
        return self.evaluationFunction(state), None
      elif agent == 0:
        return self.getMaxValue(state, agent, depth)
      else:
        return self.getProbValue(state, agent, depth)

    def getMaxValue(self, state, agent, depth):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

      if len(successorStates) == 0:
        return self.evaluationFunction(state), None
      else:
        finalValue = -sys.maxint, None
        for state, action in successorStates:
          value = self.getValue(state, nextAgent, nextDepth)
          if value[0] > finalValue[0]:
            finalValue = value[0], action
        return finalValue

    def getProbValue(self, state, agent, depth):
      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth + 1 if agent + 1 == state.getNumAgents() else depth
      successorStates = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

      if len(successorStates) == 0:
        return self.evaluationFunction(state), None
      else:
        values = []
        for state, action in successorStates:
          value = self.getValue(state, nextAgent, nextDepth)
          values.append(value)

        avg = 0.0
        for x in range(len(values)):
          avg += values[x][0]
        return avg / len(values), None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: score, number of foods left, number of capsules left, closest food, closest capsule, closest ghost, closest scaried ghost
    """
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    score = scoreEvaluationFunction(currentGameState)
    closestGhostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

    foods = currentGameState.getFood().asList()
    numFoods = len(foods)
    closestFoodDist = 0
    if foods:
      closestFoodDist = min([manhattanDistance(newPos, food) for food in foods])

    capsules = currentGameState.getCapsules()
    numCapsules = len(capsules)
    closestCapsuleDist = 0
    if capsules:
      closestCapsuleDist = min([manhattanDistance(newPos, capsule) for capsule in capsules])

    eval = score
    eval -= numFoods * 5
    eval -= numCapsules * 50
    eval -= closestFoodDist
    eval -= closestCapsuleDist
    eval += closestGhostDist
    eval -= closestGhostDist if newScaredTimes[0] > 0 else 0

    return eval

# Abbreviation
better = betterEvaluationFunction

