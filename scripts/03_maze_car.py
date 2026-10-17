"""
03_maze_car.py — 小车走迷宫: "规划 + 控制" 两层架构
====================================================
highway 程序的大脑是纯"反应式": 看一眼当前路况 → 立刻决策, 走一步看一步。
迷宫则不同 —— 必须先想清楚整条路线, 再沿路线开车。
这正是真实自动驾驶的标准两层架构:

    第一层 · 全局规划(导航): 我该走哪条路?      → BFS 广度优先搜索
    第二层 · 局部控制(开车): 沿路怎么打方向盘?  → 逐点路径跟踪

小车本身用"运动学自行车模型" —— 真实汽车不能横着走,
转弯半径由轴距和前轮转角决定, 这是所有车辆运动建模的起点。

附一次真实的失败(第一版): 我们最初用经典的"纯跟踪Pure Pursuit"控制器,
它盯着前方0.6米处的路径点开 — 但在迷宫直角弯, 前视点会"看穿墙角"
落在墙对角, 小车径直撞墙卡死(90秒压墙957帧)。
这就是教科书上的"纯跟踪切弯"缺陷, 也是为什么真实系统
规划层要输出"平滑可跟踪"的路径, 而不只是格子中心连线。

视频三段式:
    1) 蓝色扩散 = BFS 正在探索迷宫(死胡同也会被探到)
    2) 绿色亮线 = 找到的最短路径
    3) 橙色小车 = 逐点跟踪控制器沿路径行驶(弯道自动减速, 直道加速)

运行: .venv\\Scripts\\python scripts\\03_maze_car.py
产出: runs/maze_demo.mp4 (runs/ 不进 git, 只存代码)
"""
import collections
import math
import os
import random

import imageio.v2 as imageio
import numpy as np
import pygame

# ---------------- 迷宫设置 ----------------
COLS, ROWS = 21, 11                     # 格子数(迷宫生成算法要求奇数)
CELL = 26                               # 每格边长(像素)
MARGIN = 13
W, H = COLS * CELL + 2 * MARGIN, ROWS * CELL + 2 * MARGIN
S, G = (1, 1), (ROWS - 2, COLS - 2)     # 起点格/终点格 (行, 列)
SEED = 7                                # 固定种子 → 每次生成同一个迷宫(可复现)


def gen_maze(seed=SEED):
    """随机深度优先算法生成"完美迷宫": 任意两点之间恰好一条路(保证有解)"""
    rng = random.Random(seed)
    grid = [[1] * COLS for _ in range(ROWS)]   # 1=墙, 0=通路
    stack = [S]
    grid[S[0]][S[1]] = 0
    while stack:
        r, c = stack[-1]
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        carved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= ROWS - 2 and 1 <= nc <= COLS - 2 and grid[nr][nc] == 1:
                grid[nr][nc] = 0            # 打通目标格
                grid[r + dr // 2][c + dc // 2] = 0  # 打通中间的墙
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()                     # 走到头了, 回退(这就是深度优先)
    return grid


def bfs(grid, start, goal):
    """BFS: 像"洪水扩散"一样由近到远探索, 首次到达即最短。
    故意探索完整个迷宫(不提前停), 好让你在视频里看到死胡同也被探查。
    返回 (最短路径, 探索顺序)"""
    q = collections.deque([start])
    prev = {start: None}
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < ROWS and 0 <= nxt[1] < COLS
                    and grid[nxt[0]][nxt[1]] == 0 and nxt not in prev):
                prev[nxt] = cur
                q.append(nxt)
    path, node = [], goal
    while node is not None:                 # 从终点沿"来路"回溯出路径
        path.append(node)
        node = prev[node]
    return path[::-1], order


# ---------------- 第二层: 控制器 ----------------
class Car:
    """运动学自行车模型: 真实汽车不能横着移动, 转弯靠前后轮几何关系"""

    def __init__(self, x, y, th):
        self.x, self.y, self.th, self.v = x, y, th, 0.0
        self.L = 0.2                        # 轴距(米), 决定转弯灵活性

    def step(self, steer, acc, dt):
        self.th += self.v / self.L * math.tan(steer) * dt   # 转向改变车头朝向
        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self.v = max(0.0, self.v + acc * dt)


def wrap(a):
    """把角度折回 [-pi, pi]"""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def follow(car, wpts, idx):
    """逐点路径跟踪(比纯跟踪更保守, 适合栅格路径):
    目标 = 下一个还没到达的路径点; 车头偏差越大 → 方向盘打越多 + 车速越慢。
    大角度转弯时先减速再转, 就像人类司机过直角弯。"""
    while idx < len(wpts) - 1 and math.hypot(
            wpts[idx][0] - car.x, wpts[idx][1] - car.y) < 0.35:
        idx += 1
    tx, ty = wpts[idx]
    alpha = wrap(math.atan2(ty - car.y, tx - car.x) - car.th)
    steer = max(-0.65, min(0.65, 2.5 * alpha))     # 比例控制: 偏差大→打角大
    if abs(alpha) > 0.6:                           # 需要大转向 → 慢速掉头
        v_target = 0.35
    elif abs(alpha) > 0.25:
        v_target = 0.9
    else:
        v_target = 1.8
    dist_end = math.hypot(wpts[-1][0] - car.x, wpts[-1][1] - car.y)
    if dist_end < 1.0:                             # 快到终点, 平滑刹停
        v_target = min(v_target, dist_end)
    acc = max(-3.0, min(2.0, 2.0 * (v_target - car.v)))
    car.step(steer, acc, DT)
    return idx


def touching_wall(grid, car):
    """检查车体5个采样点是否压到墙格(诚实汇报, 方便调参)"""
    c_, s_ = math.cos(car.th), math.sin(car.th)
    for lx, ly in ((0.35, 0), (-0.25, 0.17), (-0.25, -0.17),
                   (0.10, 0.17), (0.10, -0.17)):
        x = car.x + lx * c_ - ly * s_
        y = car.y + lx * s_ + ly * c_
        r = min(max(int(y), 0), ROWS - 1)
        c = min(max(int(x), 0), COLS - 1)
        if grid[r][c] == 1:
            return True
    return False


# ---------------- 第一层 + 渲染 ----------------
BG = (16, 18, 22); WALL = (64, 70, 80); FLOOR = (25, 28, 34)
VISITED = (38, 70, 104); PATHC = (52, 150, 84); GOALC = (62, 240, 140)
STARTC = (70, 120, 240); CARC = (255, 140, 58); TRAIL = (130, 75, 30)
TXT = (205, 210, 220)


def px(x, y):
    """世界坐标(米) → 屏幕坐标(像素)"""
    return MARGIN + x * CELL, MARGIN + y * CELL


def draw(screen, font, grid, visited_n, show_path, wpts, car, trail,
         phase, wall_hits, t):
    screen.fill(BG)
    for r in range(ROWS):
        for c in range(COLS):
            rect = (MARGIN + c * CELL + 1, MARGIN + r * CELL + 1,
                    CELL - 2, CELL - 2)
            screen.fill(WALL if grid[r][c] else FLOOR, rect)
    for r, c in VISITED_ORDER[:visited_n]:          # BFS 扩散过程
        rect = (MARGIN + c * CELL + 1, MARGIN + r * CELL + 1,
                CELL - 2, CELL - 2)
        screen.fill(VISITED, rect)
    sr, sc = S
    screen.fill(STARTC, (MARGIN + sc * CELL + 5, MARGIN + sr * CELL + 5,
                         CELL - 10, CELL - 10))
    gr, gc = G
    screen.fill(GOALC, (MARGIN + gc * CELL + 5, MARGIN + gr * CELL + 5,
                        CELL - 10, CELL - 10))
    if show_path:
        pts = [px(x, y) for x, y in wpts]
        pygame.draw.lines(screen, PATHC, False, pts, 4)
    for x, y in trail:
        pygame.draw.circle(screen, TRAIL, (int(px(x, y)[0]), int(px(x, y)[1])), 2)
    if car is not None:                             # 小车 = 朝向三角
        c_, s_ = math.cos(car.th), math.sin(car.th)

        def body(lx, ly):
            return px(car.x + lx * c_ - ly * s_, car.y + lx * s_ + ly * c_)

        pygame.draw.polygon(screen, CARC, [
            body(0.35, 0), body(-0.25, 0.17), body(-0.25, -0.17)])
    hud = f"{phase}   t={t:4.1f}s   v={0 if car is None else car.v:.1f}m/s   " \
          f"蹭墙={wall_hits}"
    screen.blit(font.render(hud, True, TXT), (MARGIN + 4, 4))


DT = 1 / 15          # 物理步长 = 视频帧间隔
FPS = 15

if __name__ == "__main__":
    grid = gen_maze()
    path, order = bfs(grid, S, G)
    VISITED_ORDER = order
    wpts = [(c + 0.5, r + 0.5) for r, c in path]    # 格子中心 = 路径点(米)
    print(f"迷宫 {ROWS}x{COLS} | BFS探索 {len(order)} 格 | "
          f"最短路径 {len(path)} 格 | 多探的死胡同 {len(order) - len(path)} 格")

    os.environ["SDL_VIDEODRIVER"] = "dummy"         # 不弹窗, 离屏渲染
    pygame.init()
    screen = pygame.Surface((W, H))
    font = pygame.font.SysFont("microsoftyahei", 14)

    car = Car(*wpts[0], math.atan2(wpts[1][1] - wpts[0][1],
                                    wpts[1][0] - wpts[0][0]))
    frames, trail, wall_hits = [], [], 0

    def snap(visited_n=0, show_path=False, car_=None, phase="", t=0.0):
        draw(screen, font, grid, visited_n, show_path, wpts, car_,
             trail, phase, wall_hits, t)
        frames.append(np.transpose(
            pygame.surfarray.array3d(screen), (1, 0, 2)))

    # --- 第1段: BFS 探索扩散(约4秒) ---
    step = max(1, len(order) // 60)
    n = 0
    while n < len(order):
        n = min(len(order), n + step)
        snap(visited_n=n, phase="规划: BFS扩散探索迷宫...")
    snap(visited_n=len(order), phase="规划: BFS扩散探索迷宫...")
    for _ in range(25):                             # 停顿, 看清全貌
        snap(visited_n=len(order), phase="规划: BFS扩散探索迷宫...")

    # --- 第2段: 亮出最短路径 ---
    for _ in range(30):
        snap(visited_n=len(order), show_path=True,
             car_=car, phase="规划完成! 绿线=最短路径")

    # --- 第3段: 逐点跟踪驾驶 ---
    idx, t = 1, 0.0
    while t < 60:
        idx = follow(car, wpts, idx)
        t += DT
        wall_hits += touching_wall(grid, car)
        if len(frames) % 3 == 0:
            trail.append((car.x, car.y))
        dist_end = math.hypot(wpts[-1][0] - car.x, wpts[-1][1] - car.y)
        snap(visited_n=len(order), show_path=True, car_=car,
             phase="控制: 逐点路径跟踪行驶中", t=t)
        if dist_end < 0.3 and car.v < 0.15:
            break
    for _ in range(40):
        snap(visited_n=len(order), show_path=True, car_=car,
             phase=f"到达终点! 用时{t:.1f}s", t=t)

    os.makedirs("runs", exist_ok=True)
    imageio.mimwrite("runs/maze_demo.mp4", frames, fps=FPS)
    print(f"=== 完成: runs/maze_demo.mp4 ({len(frames)/FPS:.0f}秒) ===")
    print(f"行驶 {t:.1f}s | 压墙 {wall_hits} 帧 | "
          f"路径 {len(path)} 格 | 平均 {len(path)/max(t,0.01):.1f} 格/秒")
