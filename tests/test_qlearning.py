from maze_ai.maze import FREE, Maze
from maze_ai.qlearning import BUMP_PENALTY, QLearningAgent


def open_maze(n):
    return Maze([[FREE] * n for _ in range(n)], (0, 0), (n - 1, n - 1))


def test_wall_bump_keeps_state():
    agent = QLearningAgent(open_maze(3))
    s, r, done = agent.step((0, 0), 0)   # жоғары = тор шегінен тыс
    assert s == (0, 0) and r == BUMP_PENALTY and not done


def test_agent_learns_to_reach_goal():
    m = open_maze(5)
    agent = QLearningAgent(m, seed=1)
    agent.train(1500)
    path = agent.greedy_path()
    assert path is not None and path[-1] == m.goal
