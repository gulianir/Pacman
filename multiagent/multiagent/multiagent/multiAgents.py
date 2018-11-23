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
from pip.status_codes import SUCCESS
from Tkconstants import CURRENT

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
        newFood = successorGameState.getFood()
        oldFoodList = currentGameState.getFood().asList();
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        if successorGameState.isWin():
            return 99999;
        if successorGameState.isLose():
            return -99999;
        
        foodDistances = []
        for foodPos in oldFoodList:
            foodDistances.append(manhattanDistance(newPos, foodPos));
        
        closestFoodDist = min(foodDistances);
        if closestFoodDist is 0:
            return 999;
        
        ghostPositions = [];
        for ghost in newGhostStates:
            ghostPositions.append(ghost.getPosition());
            
        ghostDistances = [];
        for ghostPos in ghostPositions:
            ghostDistances.append(manhattanDistance(ghostPos, newPos));
        
        closestGhostDist = min(ghostDistances);
        
        
        return 1.0/closestFoodDist - 1.0/closestGhostDist; 
        

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
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0);
        
        scores = [];
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action);
            agent = 1%gameState.getNumAgents();
            depth = self.depth*gameState.getNumAgents()-1;
            scores.append(self.minimax(state,agent,depth));
            
        bestScore = max(scores);
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)        
        return legalMoves[chosenIndex];
        
    def minimax(self, state, agent, depth):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state);
        if agent == 0:
            # pacman!
            return self.minMaxHelper(state, agent, depth, 1);
        else:
            # adversary >:(
            return self.minMaxHelper(state, agent, depth, -1);
        
    def minMaxHelper(self, state, agent, depth, code):
        values = [];
        actions = state.getLegalActions(agent);
        for action in actions:
            state_t = state.generateSuccessor(agent, action);
            agent_t = (agent + 1) % state.getNumAgents();
            values.append(self.minimax(state_t, agent_t, depth-1)); 
        if code is 1:
            return max(values);
        return min(values);

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0);
        alpha = float('-inf');
        beta = float('inf');
        scores = [];
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action);
            agent = 1%gameState.getNumAgents();
            depth = self.depth*gameState.getNumAgents()-1;
            scores.append(self.alphabeta(state,agent,depth,alpha,beta));
            alpha = max(scores);
            
        bestScore = max(scores);
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)        
        return legalMoves[chosenIndex];
        
    def alphabeta(self, state, agent, depth, alpha, beta):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state);
        if agent == 0:
            # pacman!
            return self.minMaxHelper(state, agent, depth, 1, alpha, beta);
        else:
            # adversary >:(
            return self.minMaxHelper(state, agent, depth, -1, alpha, beta);
        
    def minMaxHelper(self, state, agent, depth, code, alpha, beta):
        values = [];
        actions = state.getLegalActions(agent);
        for action in actions:
            state_t = state.generateSuccessor(agent, action);
            agent_t = (agent + 1) % state.getNumAgents();
            values.append(self.alphabeta(state_t, agent_t, depth-1, alpha, beta));
            
            if code is 1:
                maxVal = max(values);
                if maxVal > beta:
                    return maxVal;
                if maxVal > alpha:
                    alpha = maxVal;
            else:
                minVal = min(values);
                if minVal < alpha:
                    return minVal;
                if minVal < beta:
                    beta = minVal;
                    
        if code is 1:
            return max(values);
        return min(values);

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
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0);
        
        scores = [];
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action);
            agent = 1%gameState.getNumAgents();
            depth = self.depth*gameState.getNumAgents()-1;
            scores.append(self.expectimax(state,agent,depth));
            
        bestScore = max(scores);
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)        
        return legalMoves[chosenIndex];
        
    def expectimax(self, state, agent, depth):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state);
        if agent == 0:
            # pacman!
            return self.minMaxHelper(state, agent, depth, 1);
        else:
            # adversary >:(
            return self.minMaxHelper(state, agent, depth, -1);
        
    def minMaxHelper(self, state, agent, depth, code):
        values = [];
        actions = state.getLegalActions(agent);
        for action in actions:
            state_t = state.generateSuccessor(agent, action);
            agent_t = (agent + 1) % state.getNumAgents();
            values.append(self.expectimax(state_t, agent_t, depth-1)); 
        if code is 1:
            return max(values);
        return (sum(values) + 0.0)/len(values);

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Incorporated current game score and lessened weight of ghost distances
                   Otherwise, same as part 1. 
    """
    "*** YOUR CODE HERE ***"
    
    oldFoodList = currentGameState.getFood().asList();
    currentPos = currentGameState.getPacmanPosition();
    currentGhostStates = currentGameState.getGhostStates();
        
    foodDistances = []
    for foodPos in oldFoodList:
        foodDistances.append(manhattanDistance(currentPos, foodPos));
        
    if (not len(foodDistances) == 0):
        closestFoodDist = min(foodDistances);
    else:
        closestFoodDist = 0;
        
    ghostPositions = [];
    for ghost in currentGhostStates:
        ghostPositions.append(ghost.getPosition());
            
    ghostDistances = [];
    for ghostPos in ghostPositions:
        ghostDistances.append(manhattanDistance(ghostPos, currentPos) ** (1.0/2));
        
    closestGhostDist = min(ghostDistances);
        
    return currentGameState.getScore() + closestGhostDist - closestFoodDist; 


# Abbreviation
better = betterEvaluationFunction

