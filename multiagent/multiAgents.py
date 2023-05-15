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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition() # 当前吃豆人位置
        newFood = successorGameState.getFood() # 当前食物位置
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        GhostPos=successorGameState.getGhostPositions() # 获取幽灵位置
        ghostdist=min([(abs(each[0] - newPos[0]) + abs(each[1] - newPos[1])) for each in GhostPos])  # 计算吃豆人行动后到幽灵的距离
        if ghostdist<=4 and ghostdist!=0:
            ghostscore=-15/ghostdist # 计算幽灵惩罚函数
        else:
            ghostscore=0
        foodaround=[]
        Width=newFood.width
        Height=newFood.height
        for i in range(Width):
            for j in range(Height):
                if newFood[i][j]==1:
                    foodaround.append((i,j))    # 暴力枚举每一个存在食物的地方
        if ghostdist>=2 and len(foodaround)>0:
            nearestfooddis=min([manhattanDistance(newPos,food) for food in foodaround ]) # 计算距离吃豆人最近的食物距离
            foodscore=10/nearestfooddis # 计算食物评估函数
        else:
            foodscore=0
        currentscore=successorGameState.getScore()
        newscore=currentscore+foodscore+ghostscore
        return newscore

def scoreEvaluationFunction(currentGameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # ghost可能不止一个
        GhostIndex = range(1, gameState.getNumAgents())

        # 目标状态：游戏结束或者搜索到一定深度
        def targetstate(state, Depth):
            return state.isWin() or state.isLose() or Depth == self.depth

        # ghost是min玩家
        def min_value(state, Depth, ghost):  # minimizer

            if targetstate(state, Depth):
                return self.evaluationFunction(state)

            MIN = float("inf")  # MIN初始值为无穷大
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:  # 递归的查找最小值，如果最后一个ghost已经作出行动了，下一次便是轮到pacman
                    MIN = min(MIN, max_value(state.generateSuccessor(ghost, action), Depth + 1))
                else:  # 否则就遍历所有ghost，每一个ghost作出行动后再由pacman行动
                    MIN = min(MIN, min_value(state.generateSuccessor(ghost, action), Depth, ghost + 1))
            return MIN
        # pacman是max玩家，每次选择效用值最大的行动
        def max_value(state, Depth):

            if targetstate(state, Depth):
                return self.evaluationFunction(state)

            MAX = -float("inf")  # MAX初始值为无穷小
            for action in state.getLegalActions(0):
                # 递归查找下一个状态中效用值最大的
                MAX = max(MAX, min_value(state.generateSuccessor(0, action), Depth, 1))
            return MAX

        ans = [(action, min_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        ans.sort(key=lambda k: k[1])

        return ans[-1][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # ghost可能不止一个
        GhostIndex = range(1, gameState.getNumAgents())

        # 目标状态：游戏结束或者搜索到一定深度
        def targetstate(state, Depth):
            return state.isWin() or state.isLose() or Depth == self.depth

        # ghost是min玩家
        def min_value(state, Depth, ghost,A,B):

            if targetstate(state, Depth):
                return self.evaluationFunction(state)

            MIN = float("inf")  # MIN初始值为无穷大
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:  # 递归的查找最小值，如果最后一个ghost已经作出行动了，下一次便是轮到pacman
                    MIN = min(MIN, max_value(state.generateSuccessor(ghost, action), Depth + 1,A,B))
                else:  # 否则就遍历所有ghost，每一个ghost作出行动后再由pacman行动
                    MIN = min(MIN, min_value(state.generateSuccessor(ghost, action), Depth, ghost + 1,A,B))
                if MIN < A:  # 剪枝
                    return MIN
                B = min(B, MIN)
            return MIN

        # pacman是max玩家，每次选择效用值最大的行动
        def max_value(state, Depth,A,B):

            if targetstate(state, Depth):
                return self.evaluationFunction(state)

            MAX = -float("inf")  # MAX初始值为无穷小
            for action in state.getLegalActions(0):
                # 递归查找下一个状态中效用值最大的
                MAX = max(MAX, min_value(state.generateSuccessor(0, action), Depth, 1,A,B))
                if MAX > B:
                  return MAX
                A = max(A, MAX)
            return MAX


        def alphabeta(state):

            MAX = -float("inf")
            act = None
            A = -float("inf")   # A为目前已经发现的MAX的极大值，如果搜索过程中发现
            B = float("inf")

            for action in state.getLegalActions(0):  # 求最大值
                value = min_value(gameState.generateSuccessor(0, action), 0, 1, A, B)

                if MAX< value:
                    MAX = value
                    act = action

                if MAX > B:  # 剪枝
                    return MAX
                A = max(A, value)
            return act

        return alphabeta(gameState)

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
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]

        # 目标状态：游戏结束或者搜索到一定深度
        def target(state, d):
            return state.isWin() or state.isLose() or d == self.depth

        # ghost不一定每次作出最好决定，要计算期望值
        def exp_value(state, d, ghost):  # minimizer

            if target(state, d):
                return self.evaluationFunction(state)

            v = 0
            prob = 1 / len(state.getLegalActions(ghost))# 每种情况的概率

            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:# 递归的查找期望值，如果最后一个ghost已经作出行动了，下一次便是轮到pacman
                    v += prob * max_value(state.generateSuccessor(ghost, action), d + 1)
                else:# 否则就遍历所有ghost，每一个ghost作出行动后再由pacman行动
                    v += prob * exp_value(state.generateSuccessor(ghost, action), d, ghost + 1)

            return v

        def max_value(state, d):  # maximizer

            if target(state, d):
                return self.evaluationFunction(state)

            MAX = -float("inf")
            for action in state.getLegalActions(0):
                MAX = max(MAX, exp_value(state.generateSuccessor(0, action), d, 1))
            return MAX

        ans = [(action, exp_value(gameState.generateSuccessor(0, action), 0, 1)) for action in
               gameState.getLegalActions(0)]
        ans.sort(key=lambda k: k[1])

        return ans[-1][0]



def betterEvaluationFunction(currentGameState:GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    Pos = currentGameState.getPacmanPosition()  # current position
    Food = currentGameState.getFood()  # current food
    GhostStates = currentGameState.getGhostStates()  # ghost state
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    capsulepos=currentGameState.getCapsules()

    # 食物带来的积极影响
    if len(Food.asList()) > 0:
        nearestFood = (min([manhattanDistance(Pos, food) for food in Food.asList()]))
        foodScore = 10 / nearestFood
    else:
        foodScore = 0
    if len(capsulepos) > 0:
        nearestcapsule = (min([manhattanDistance(Pos, capsule) for capsule in capsulepos]))
        capsuleScore = 0.5 / nearestcapsule
    else:
        capsuleScore = 0
    # 幽灵带来的负面影响
    nearestGhost = min([manhattanDistance(Pos, ghostState.configuration.pos) for ghostState in GhostStates])
    if nearestGhost<4 and nearestGhost!=0:
        dangerScore = -11 / nearestGhost
    else:
        dangerScore=0
    # 幽灵剩下能被吃掉的时间
    totalScaredTimes = sum(ScaredTimes)
    # return sum of all value
    return currentGameState.getScore() + foodScore + dangerScore + totalScaredTimes+capsuleScore




# Abbreviation
better = betterEvaluationFunction
