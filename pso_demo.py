"""
PSO Demo: Function Optimization + Robot Path Planning
智能算法课程实验 — 粒子群优化算法（Particle Swarm Optimization）

运行方式：python pso_demo.py
关闭第一个动画窗口后自动弹出第二个。
如需保存 gif，把 SAVE_GIF 改为 True。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SAVE_GIF = False          # 改成 True 会额外保存 gif 文件
FPS = 15
N_ITER = 100
N_PARTICLES = 50

# ============================================================
# PSO 核心
# ============================================================
class PSO:
    def __init__(self, dim, n, bounds, fitness, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, v_frac=0.2):
        self.dim = dim
        self.n = n
        self.bounds = np.array(bounds, dtype=float)          # (dim, 2)
        self.fitness = fitness
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2

        self.X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (n, dim))
        v_range = (self.bounds[:, 1] - self.bounds[:, 0]) * v_frac
        self.V = np.random.uniform(-v_range, v_range, (n, dim))

        self.pbest = self.X.copy()
        self.pbest_val = np.array([fitness(x) for x in self.X])
        g = int(np.argmin(self.pbest_val))
        self.gbest = self.pbest[g].copy()
        self.gbest_val = self.pbest_val[g]

        self.history = [self.gbest_val]

    def step(self, it, max_it):
        w = self.w_max - (self.w_max - self.w_min) * it / max_it
        r1 = np.random.rand(self.n, self.dim)
        r2 = np.random.rand(self.n, self.dim)
        self.V = w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (self.gbest - self.X)
        self.X = self.X + self.V
        for d in range(self.dim):
            self.X[:, d] = np.clip(self.X[:, d], self.bounds[d, 0], self.bounds[d, 1])

        vals = np.array([self.fitness(x) for x in self.X])
        upd = vals < self.pbest_val
        self.pbest[upd] = self.X[upd]
        self.pbest_val[upd] = vals[upd]
        g = int(np.argmin(self.pbest_val))
        if self.pbest_val[g] < self.gbest_val:
            self.gbest_val = self.pbest_val[g]
            self.gbest = self.pbest[g].copy()
        self.history.append(self.gbest_val)


# ============================================================
# Part A: 经典测试函数
# ============================================================
def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

FUNCS = [
    (sphere,     "Sphere (unimodal)",    (-5.12, 5.12)),
    (rastrigin,  "Rastrigin (multimodal)", (-5.12, 5.12)),
    (rosenbrock, "Rosenbrock (narrow valley)", (-2.0, 2.0)),
]


def run_function_demo():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("PSO on Classical Benchmark Functions", fontsize=14, fontweight="bold")

    pso_list, scatter_list, line_list, cont_list = [], [], [], []

    for col, (f, name, (lo, hi)) in enumerate(FUNCS):
        ax = axes[0, col]
        g = np.linspace(lo, hi, 120)
        Gx, Gy = np.meshgrid(g, g)
        Gz = np.array([[f(np.array([a, b])) for a in g] for b in g])
        cont = ax.contourf(Gx, Gy, Gz, levels=20, cmap="viridis", alpha=0.7)
        ax.contour(Gx, Gy, Gz, levels=10, colors="white", linewidths=0.3, alpha=0.5)
        scatter, = ax.plot([], [], "w.", markersize=6, label="particles")
        gbest_pt, = ax.plot([], [], "r*", markersize=15, label="gbest")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(name); ax.legend(loc="upper right", fontsize=8)

        axc = axes[1, col]
        axc.set_title(f"{name} - convergence")
        axc.set_xlabel("iteration"); axc.set_ylabel("best fitness")
        axc.set_xlim(0, N_ITER); axc.set_yscale("log")
        conv_line, = axc.plot([], [], "r-", lw=1.5)

        pso = PSO(2, N_PARTICLES, [(lo, hi), (lo, hi)], f)
        pso_list.append(pso)
        scatter_list.append((scatter, gbest_pt))
        line_list.append(conv_line)

    def update(frame):
        for pso in pso_list:
            pso.step(frame, N_ITER)
        for i, pso in enumerate(pso_list):
            scatter, gbest_pt = scatter_list[i]
            scatter.set_data(pso.X[:, 0], pso.X[:, 1])
            gbest_pt.set_data([pso.gbest[0]], [pso.gbest[1]])
            line_list[i].set_data(range(len(pso.history)), pso.history)
        return []

    ani = FuncAnimation(fig, update, frames=N_ITER, interval=1000 // FPS, blit=False)
    plt.tight_layout()
    if SAVE_GIF:
        ani.save("pso_function.gif", writer=PillowWriter(fps=FPS))
    plt.show()


# ============================================================
# Part B: 栅格地图路径规划
# ============================================================
GRID_N = 20
N_WP = 8
START = (0.0, 0.0)
END = (GRID_N - 1, GRID_N - 1)

def build_grid(seed=42):
    rng = np.random.default_rng(seed)
    grid = np.zeros((GRID_N, GRID_N), dtype=int)
    keep_clear = {(0, 0), (1, 0), (0, 1),
                  (GRID_N - 1, GRID_N - 1), (GRID_N - 2, GRID_N - 1), (GRID_N - 1, GRID_N - 2)}
    placed = 0
    while placed < 45:
        i = rng.integers(0, GRID_N)
        j = rng.integers(0, GRID_N)
        if (i, j) in keep_clear:
            continue
        if grid[i, j] == 0:
            grid[i, j] = 1
            placed += 1
    return grid

def seg_hits_obstacle(p0, p1, grid, n=20):
    t = np.linspace(0, 1, n)
    xs = p0[0] + t * (p1[0] - p0[0])
    ys = p0[1] + t * (p1[1] - p0[1])
    ix = np.clip(np.round(xs).astype(int), 0, grid.shape[0] - 1)
    iy = np.clip(np.round(ys).astype(int), 0, grid.shape[1] - 1)
    return bool(np.any(grid[ix, iy] == 1))

def make_path_fitness(grid):
    def fitness(particle):
        pts = np.zeros((N_WP + 2, 2))
        pts[0] = START
        pts[-1] = END
        pts[1:-1, 0] = particle[0::2]
        pts[1:-1, 1] = particle[1::2]
        length = np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))
        pen = 0
        for i in range(len(pts) - 1):
            if seg_hits_obstacle(pts[i], pts[i + 1], grid):
                pen += 500
        return length + pen
    return fitness

def path_from_particle(particle):
    xs, ys = [START[0]], [START[1]]
    for i in range(N_WP):
        xs.append(particle[2 * i])
        ys.append(particle[2 * i + 1])
    xs.append(END[0]); ys.append(END[1])
    return xs, ys


def run_path_demo():
    grid = build_grid()
    fitness = make_path_fitness(grid)
    bounds = [(0, GRID_N - 1)] * (2 * N_WP)
    pso = PSO(2 * N_WP, N_PARTICLES, bounds, fitness, v_frac=0.1)

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("PSO for Robot Path Planning on 20x20 Grid", fontsize=14, fontweight="bold")

    ax.set_xlim(-0.5, GRID_N - 0.5)
    ax.set_ylim(-0.5, GRID_N - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for i in range(GRID_N):
        for j in range(GRID_N):
            if grid[i, j] == 1:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="gray"))
    ax.plot(*START, "go", markersize=12, label="start")
    ax.plot(*END, "r*", markersize=18, label="goal")
    all_paths_lines = []
    for _ in range(N_PARTICLES):
        line, = ax.plot([], [], "-", color="lightblue", lw=0.5, alpha=0.3)
        all_paths_lines.append(line)
    best_line, = ax.plot([], [], "r-", lw=2.5, label="best path")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Map (gray=obstacle)")

    axc.set_xlim(0, N_ITER); axc.set_ylim(0, 60)
    axc.set_xlabel("iteration"); axc.set_ylabel("best path length (+ penalty)")
    axc.set_title("Convergence")
    conv, = axc.plot([], [], "b-", lw=1.5)
    info = axc.text(0.02, 0.95, "", transform=axc.transAxes, va="top", fontsize=10)

    def update(frame):
        pso.step(frame, N_ITER)
        for i in range(N_PARTICLES):
            xs, ys = path_from_particle(pso.X[i])
            all_paths_lines[i].set_data(xs, ys)
        bxs, bys = path_from_particle(pso.gbest)
        best_line.set_data(bxs, bys)
        conv.set_data(range(len(pso.history)), pso.history)
        info.set_text(f"iter={frame}\nbest={pso.gbest_val:.2f}")
        return []

    ani = FuncAnimation(fig, update, frames=N_ITER, interval=1000 // FPS, blit=False)
    plt.tight_layout()
    if SAVE_GIF:
        ani.save("pso_path.gif", writer=PillowWriter(fps=FPS))
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    run_function_demo()
    run_path_demo()
