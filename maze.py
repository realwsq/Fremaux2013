import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# -----------------------------
# Maze geometry (square + U obstacle + goal)
# -----------------------------
@dataclass
class Maze:
    half_size: float = 10.0  # square is 20x20 centered at origin
    goal_r: float = 1.0      # goal radius
    dx_bounce: float = 0.1   # bounce distance

    # U-shaped obstacle: 3 segments (width 2, length 10), around goal
    obs_width: float = 2.0
    obs_len: float = 10.0

    Robst: float = -0.6
    Rgoal: float = 90.0
    
    x_goal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def goal_reached(self, x):
        return ((x[0] - self.x_goal[0])**2 + (x[1] - self.x_goal[1])**2) <= self.goal_r**2

    def obstacle_hit_and_normal(self, x):
        """
        Return (hit, inward_normal). The U is centered at origin, open toward +y.
        U segments:
          left vertical bar:  x ∈ [-L/2-w/2, -L/2+w/2], y ∈ [-L/2, +L/2]
          right vertical bar: x ∈ [+L/2-w/2, +L/2+w/2], y ∈ [-L/2, +L/2]
          bottom bar:         x ∈ [-L/2, +L/2], y ∈ [-L/2-w/2, -L/2+w/2]
        """
        w = self.obs_width
        L = self.obs_len

        # Left vertical bar (at x ≈ -5)
        if (-L/2 - w/2 <= x[0] <= -L/2 + w/2) and (-L/2 <= x[1] <= L/2):
            # Normal points away from obstacle center
            if x[0] < -L/2:
                return True, np.array([-1.0, 0.0])
            else:
                return True, np.array([+1.0, 0.0])

        # Right vertical bar (at x ≈ +5)
        if (L/2 - w/2 <= x[0] <= L/2 + w/2) and (-L/2 <= x[1] <= L/2):
            if x[0] > L/2:
                return True, np.array([+1.0, 0.0])
            else:
                return True, np.array([-1.0, 0.0])

        # Bottom horizontal bar (at y ≈ -5)
        if (-L/2 <= x[0] <= L/2) and (-L/2 - w/2 <= x[1] <= -L/2 + w/2):
            if x[1] < -L/2:
                return True, np.array([0.0, -1.0])
            else:
                return True, np.array([0.0, +1.0])

        return False, None

    def wall_hit_and_normal(self, x):
        """
        If outside square, return (hit, inward_normal).
        """
        hs = self.half_size
        if x[0] < -hs:
            return True, np.array([+1.0, 0.0])
        if x[0] > +hs:
            return True, np.array([-1.0, 0.0])
        if x[1] < -hs:
            return True, np.array([0.0, +1.0])
        if x[1] > +hs:
            return True, np.array([0.0, -1.0])
        return False, None

    def step(self, x, a, dt):
        """
        Agent dynamics Eq. 21: xdot = a if inside; else bounce back by dx along inward normal.
        Also punish on obstacle/wall touch with Robst=-1 (delivered as event R).
        """
        x_next = x + dt * a

        # Obstacle collision
        hit_obs, n_obs = self.obstacle_hit_and_normal(x_next)
        if hit_obs:
            x_next = x + self.dx_bounce * n_obs
            return x_next, self.Robst

        # Wall collision
        hit_wall, n_wall = self.wall_hit_and_normal(x_next)
        if hit_wall:
            x_next = x + self.dx_bounce * n_wall
            return x_next, self.Robst

        return x_next, 0.0

    def plot_map(self, ax=None, plot_obstacles=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 9))

        """Plot maze elements on given axis."""
        # Draw maze elements
        # Goal (green circle)
        goal_circle = plt.Circle(self.x_goal, self.goal_r, color='lime', alpha=0.8,
                                linewidth=2, fill=False, label='Goal')
        ax.add_patch(goal_circle)

        if plot_obstacles:
            # U-shaped obstacle (red)
            L = self.obs_len
            w = self.obs_width
            # Left bar
            ax.add_patch(plt.Rectangle((-L/2 - w/2, -L/2), w, L, color='red', alpha=0.6, label='Obstacle'))
            # Right bar
            ax.add_patch(plt.Rectangle((L/2 - w/2, -L/2), w, L, color='red', alpha=0.6))
            # Bottom bar
            ax.add_patch(plt.Rectangle((-L/2, -L/2 - w/2), L, w, color='red', alpha=0.6))

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        return ax
