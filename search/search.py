# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    from game import Directions
    close = []      # close表存放已经访问过的节点
    open = Stack()  # open表存放未拓展的节点
    action=[]     #action存放访问每一步的方向
    open.push((problem.getStartState(),[])) # 把初始节点放入open表 搜索树的节点结构为（当前状态，[actions]从初始状态到达当前状态经过的动作集合）
    while open.isEmpty()==False :
        current,action= open.pop()  # open表中弹出一个节点
        if problem.isGoalState(current) :   # 判断是否到达目标
            return action
        if current not in close:          # 如果没有访问过
            expand = problem.getSuccessors(current)  # 拓展后继节点
            close.append(current)             # 把current加入close表
            for nextnode,nextaction,cost in expand:
                newaction=action+[nextaction]
                open.push((nextnode,newaction))

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    from game import Directions
    close = []  # close表存放已经访问过的节点
    open = Queue()  # open表存放未拓展的节点
    open.push((problem.getStartState(), []))  # 把初始节点放入open表 搜索树的节点结构为（当前状态，[actions]从初始状态到达当前状态经过的动作集合）
    while open.isEmpty() == False:
        current, action = open.pop()  # open表中弹出一个节点
        if problem.isGoalState(current):  # 判断是否到达目标
            return action
        if current not in close:  # 如果没有访问过
            expand = problem.getSuccessors(current)  # 拓展后继节点
            close.append(current)  # 把current加入close表
            for nextnode, nextaction, cost in expand:
                newaction = action + [nextaction]
                open.push((nextnode, newaction))
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    visited = []  # visited表存放已经访问过的节点
    fringe = util.PriorityQueue()  # 优先队列fringe存放未拓展的节点，按耗散排序
    # 把初始节点放入fringe表 搜索树的节点结构为（当前状态，[actions]从初始状态到达当前状态经过的动作集合，当前cost）
    fringe.push((problem.getStartState(), []), 0)

    while fringe.isEmpty() == False:
         current , action = fringe.pop()
         if problem.isGoalState(current):
             return action
         if current not in visited:
            visited.append(current)
            expand=problem.getSuccessors(current)
            for nextnode,nextaction,cost in expand:
                newaction=action+[nextaction]
                newcost=problem.getCostOfActions(newaction)  # 每一步都记录访问耗散最小的节点
                fringe.push((nextnode,newaction),newcost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = []  # visited表存放已经访问过的节点
    fringe = util.PriorityQueue()  # 优先队列fringe存放未拓展的节点，按耗散排序
    # 把初始节点放入fringe表 搜索树的节点结构为（当前状态，[actions]从初始状态到达当前状态经过的动作集合，当前cost）
    fringe.push((problem.getStartState(), []), 0)

    while fringe.isEmpty() == False:
        current, action = fringe.pop()
        if problem.isGoalState(current):
            return action
        if current not in visited: # 如果没有访问过
            visited.append(current)
            expand = problem.getSuccessors(current)
            for nextnode, nextaction, cost in expand:
                newaction = action + [nextaction]
                newcost=problem.getCostOfActions(newaction)+heuristic(nextnode,problem) # 每一步代价等于行动本身代价加上启发式函数的值
                fringe.push((nextnode, newaction), newcost)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
