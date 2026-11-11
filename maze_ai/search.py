"""2-апта: BFS, DFS, UCS, A* (Manhattan эвристикасы)."""
import heapq
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SearchResult:
    name: str
    path: list          # None болса жол жоқ
    cost: int
    expanded: list      # түйіндерді қарастыру реті
    time_ms: float


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_cost(maze, path):
    return sum(maze.cost(p) for p in path[1:])


def _rebuild(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def _result(maze, name, parent, expanded, t0, found):
    path = _rebuild(parent, maze.goal) if found else None
    cost = path_cost(maze, path) if path else None
    return SearchResult(name, path, cost, expanded, (time.perf_counter() - t0) * 1000)


def bfs(maze):
    t0 = time.perf_counter()
    parent, queue, expanded = {maze.start: None}, deque([maze.start]), []
    while queue:
        cur = queue.popleft()
        expanded.append(cur)
        if cur == maze.goal:
            return _result(maze, "BFS", parent, expanded, t0, True)
        for nxt, _ in maze.neighbors(cur):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return _result(maze, "BFS", parent, expanded, t0, False)


def dfs(maze):
    """Тереңдікке іздеу (стек): жол табады, бірақ оңтайлы болуына кепілдік жоқ."""
    t0 = time.perf_counter()
    parent, stack, visited, expanded = {maze.start: None}, [maze.start], set(), []
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        expanded.append(cur)
        if cur == maze.goal:
            return _result(maze, "DFS", parent, expanded, t0, True)
        for nxt, _ in reversed(list(maze.neighbors(cur))):
            if nxt not in visited:
                parent[nxt] = cur
                stack.append(nxt)
    return _result(maze, "DFS", parent, expanded, t0, False)


def _best_first(maze, name, h):
    t0 = time.perf_counter()
    parent, g = {maze.start: None}, {maze.start: 0}
    heap, closed, expanded = [(h(maze.start), 0, maze.start)], set(), []
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded.append(cur)
        if cur == maze.goal:
            return _result(maze, name, parent, expanded, t0, True)
        for nxt, c in maze.neighbors(cur):
            ng = gc + c
            if ng < g.get(nxt, float("inf")):
                g[nxt], parent[nxt] = ng, cur
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))
    return _result(maze, name, parent, expanded, t0, False)


def ucs(maze):
    return _best_first(maze, "UCS", lambda p: 0)


def astar(maze):
    return _best_first(maze, "A*", lambda p: manhattan(p, maze.goal))


ALGORITHMS = {"BFS": bfs, "DFS": dfs, "UCS": ucs, "A*": astar}
