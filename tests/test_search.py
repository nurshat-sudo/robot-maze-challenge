from maze_ai.maze import FREE, MUD, WALL, Maze
from maze_ai.search import astar, bfs, dfs, ucs


def open_maze(n):
    return Maze([[FREE] * n for _ in range(n)], (0, 0), (n - 1, n - 1))


def test_all_find_shortest_on_open_grid():
    m = open_maze(5)
    assert bfs(m).cost == ucs(m).cost == astar(m).cost == 8


def test_no_path_returns_none():
    g = [[FREE] * 3, [WALL] * 3, [FREE] * 3]
    m = Maze(g, (0, 0), (2, 2))
    assert bfs(m).path is None and ucs(m).path is None and astar(m).path is None


def test_bfs_ignores_cost_but_ucs_astar_optimal():
    g = [[FREE, MUD, FREE], [FREE] * 3, [FREE] * 3]
    m = Maze(g, (0, 0), (0, 2))
    assert bfs(m).cost == 6
    assert ucs(m).cost == astar(m).cost == 4


def test_astar_expands_no_more_than_ucs():
    m = open_maze(10)
    assert len(astar(m).expanded) <= len(ucs(m).expanded)


def test_dfs_finds_valid_path_and_none_when_blocked():
    m = open_maze(6)
    r = dfs(m)
    assert r.path[0] == m.start and r.path[-1] == m.goal
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(r.path, r.path[1:]))
    assert r.cost >= astar(m).cost          # DFS оңтайлыдан жақсы бола алмайды
    blocked = Maze([[FREE] * 3, [WALL] * 3, [FREE] * 3], (0, 0), (2, 2))
    assert dfs(blocked).path is None
