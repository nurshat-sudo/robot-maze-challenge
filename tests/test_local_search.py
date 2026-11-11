from maze_ai.local_search import fitness, hill_climb
from maze_ai.maze import FREE, Maze
from maze_ai.search import astar


def test_hill_climb_never_gets_worse_and_stays_solvable():
    m = Maze([[FREE] * 6 for _ in range(6)], (0, 0), (5, 5))
    before = fitness(m)
    best, history = hill_climb(m, iterations=150, restarts=1, seed=3)
    assert astar(best).path is not None
    assert fitness(best) >= before
    assert history == sorted(history)
