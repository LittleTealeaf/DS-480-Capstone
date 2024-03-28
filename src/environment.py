from mazelib import Maze
from mazelib.generate.Prims import Prims


class Environment:
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    def __init__(self, height: int = 10, width: int = 10, seed=None):
        self.maze = Maze(seed)
        self.maze.generator = Prims(height, width)
        self.maze.generate()

        self.width, self.height = self.maze.grid.shape

        self.maze.generate_entrances(start_outer=False, end_outer=True)

        assert self.maze.start != None
        self.x, self.y = self.maze.start

        assert self.maze.end != None
        self.goal_x, self.goal_y = self.maze.end

    def get_observations(self):
        obs = self.maze.grid.flatten().copy()
        size = int(obs.size)
        obs.resize(size + 2 * (self.width + self.height), refcheck=False)
        obs[size + self.x] = 1
        obs[size + self.width + self.y] = 1
        obs[size + self.width + self.height + self.goal_x] = 1
        obs[size + 2 * self.width + self.height + self.goal_y] = 1
        return obs

    def get_obs_length(self):
        return int(self.get_observations().size)

    def set_position(self, x, y) -> bool:
        if self.maze.grid[y][x] == 0:
            self.x, self.y = x, y
            return True
        return False

    def get_valid_positions(self):
        assert self.maze != None
        return [
            (i // self.width, i % self.width)
            for i in range(self.width * self.height)
            if self.maze.grid[i // self.width][i % self.width] == 0
        ]

    def move(self, move: int):
        dx, dy = 0, 0
        if move == 0:
            dy = -1
        elif move == 1:
            dy = 1
        elif move == 2:
            dx = 1
        else:
            dx = -1

        nx = dx + self.x
        ny = dy + self.y

        if (
            nx >= 0
            and ny >= 0
            and nx <= self.width
            and ny <= self.height
            and self.maze.grid[ny][nx] == 0
        ):
            self.x = nx
            self.y = ny
    
    def is_solved(self) -> bool:
        return self.x == self.goal_x and self.y == self.goal_y

    def get_reward(self):
        dx = abs(self.x - self.goal_x)
        dy = abs(self.y - self.goal_y)
        return 1 / (dx + dy)

