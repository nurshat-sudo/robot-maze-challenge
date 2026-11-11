"""Лабиринт ортасы (1-апта: агент әрекет ететін орта)."""
import random
from dataclasses import dataclass

FREE, WALL, MUD = 0, 1, 2
COST = {FREE: 1, MUD: 5}                       # ұяшыққа кіру құны
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # жоғары, төмен, солға, оңға


@dataclass
class Maze:
    grid: list
    start: tuple
    goal: tuple

    @property
    def rows(self):
        return len(self.grid)

    @property
    def cols(self):
        return len(self.grid[0])

    def in_bounds(self, p):
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    def passable(self, p):
        return self.in_bounds(p) and self.grid[p[0]][p[1]] != WALL

    def cost(self, p):
        return COST[self.grid[p[0]][p[1]]]

    def neighbors(self, p):
        for dr, dc in ACTIONS:
            q = (p[0] + dr, p[1] + dc)
            if self.passable(q):
                yield q, self.cost(q)

    def copy(self):
        return Maze([r[:] for r in self.grid], self.start, self.goal)

    def validate(self):
        """Қате болса мәтін, әйтпесе None қайтарады."""
        for name, p in (("Бастау нүктесі (Start)", self.start), ("Мақсат нүктесі (Goal)", self.goal)):
            if not self.in_bounds(p):
                return f"{name} тор шегінен тыс."
            if self.grid[p[0]][p[1]] == WALL:
                return f"{name} қабырғада тұр."
        if self.start == self.goal:
            return "Бастау және мақсат нүктелері бірдей болмауы керек (Start ≠ Goal)."
        return None


def random_maze(rows, cols, wall_p=0.2, mud_p=0.1, seed=0, start=None, goal=None):
    rng = random.Random(seed)
    start = start or (0, 0)
    goal = goal or (rows - 1, cols - 1)
    grid = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            x = rng.random()
            row.append(WALL if x < wall_p else MUD if x < wall_p + mud_p else FREE)
        grid.append(row)
    for r, c in (start, goal):
        grid[r][c] = FREE
    return Maze(grid, start, goal)
