"""5-апта: Q-learning (MDP: күй=ұяшық, әрекет=4 бағыт)."""
import random

from .maze import ACTIONS

GOAL_REWARD = 100
BUMP_PENALTY = -5


class QLearningAgent:
    def __init__(self, maze, alpha=0.5, gamma=0.95, epsilon=0.2, seed=0):
        self.maze, self.alpha, self.gamma, self.epsilon = maze, alpha, gamma, epsilon
        self.rng = random.Random(seed)
        self.Q = {}

    def q(self, s):
        return self.Q.setdefault(s, [0.0] * len(ACTIONS))

    def step(self, s, a):
        """Ортаның жауабы: (жаңа күй, сыйақы, аяқталды ма)."""
        nxt = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
        if not self.maze.passable(nxt):
            return s, BUMP_PENALTY, False
        reward = -self.maze.cost(nxt)
        done = nxt == self.maze.goal
        return nxt, reward + (GOAL_REWARD if done else 0), done

    def choose(self, s):
        """ε-greedy таңдау."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        qs = self.q(s)
        best = max(qs)
        return self.rng.choice([i for i, v in enumerate(qs) if v == best])

    def train(self, episodes=800, max_steps=None):
        max_steps = max_steps or self.maze.rows * self.maze.cols * 4
        rewards, steps = [], []
        for _ in range(episodes):
            s, total, t = self.maze.start, 0, 0
            for t in range(1, max_steps + 1):
                a = self.choose(s)
                s2, r, done = self.step(s, a)
                target = r if done else r + self.gamma * max(self.q(s2))
                self.q(s)[a] += self.alpha * (target - self.q(s)[a])
                s, total = s2, total + r
                if done:
                    break
            rewards.append(total)
            steps.append(t)
        return rewards, steps

    def greedy_path(self):
        """Үйренген саясат бойынша жол; үйренбесе (цикл) None."""
        s, path, seen = self.maze.start, [self.maze.start], {self.maze.start}
        while s != self.maze.goal:
            qs = self.q(s)
            s2, _, _ = self.step(s, qs.index(max(qs)))
            if s2 in seen:
                return None
            seen.add(s2)
            path.append(s2)
            s = s2
        return path
