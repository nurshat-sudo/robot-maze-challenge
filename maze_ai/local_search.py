"""3-апта: Hill Climbing (random restart) — A*-қа ең қиын лабиринтті іздеу.
Күй = лабиринт торы; көрші = бір ұяшықты өзгерту; fitness = A* қарастырған түйін саны."""
import random

from .maze import FREE, MUD, WALL, random_maze
from .search import astar


def fitness(maze):
    r = astar(maze)
    return len(r.expanded) if r.path else -1   # жол жоқ = жарамсыз


def hill_climb(maze, iterations=300, restarts=3, seed=0):
    rng = random.Random(seed)
    best, best_f, history = None, -2, []
    for k in range(restarts):
        if k == 0:
            cur = maze.copy()
        else:
            cur = random_maze(maze.rows, maze.cols, 0.2, 0.1, rng.randrange(10**6), maze.start, maze.goal)
        cur_f = fitness(cur)
        for _ in range(iterations):
            r, c = rng.randrange(maze.rows), rng.randrange(maze.cols)
            if (r, c) not in (maze.start, maze.goal):
                cand = cur.copy()
                cand.grid[r][c] = rng.choice([v for v in (FREE, WALL, MUD) if v != cand.grid[r][c]])
                f = fitness(cand)
                if f > cur_f:           # тек қатаң жақсару қабылданады
                    cur, cur_f = cand, f
            history.append(max(best_f, cur_f))
        if cur_f > best_f:
            best, best_f = cur, cur_f
    return best, history
