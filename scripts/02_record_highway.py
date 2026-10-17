"""
02_record_highway.py — 把仿真过程录成视频
==========================================
和 01 是同一颗"大脑"(同样的规则策略)，区别只有两点:
  1. 不弹实时窗口, 而是把每一帧画面存进视频文件 → 可反复回看
  2. 决策频率从默认 5Hz 提到 15Hz → 车的反应更快, 画面更流畅

产出: runs/highway_demo.mp4 (约30秒)
注意: runs/ 目录被 .gitignore 排除 — 视频这类"实验产物"不进 git 仓库,
      git 只保管代码; 产物太大太杂, 会把仓库撑爆。

运行方式:
    .venv\\Scripts\\python scripts\\02_record_highway.py
"""
import os

import gymnasium as gym
import highway_env  # noqa: F401
import imageio.v2 as imageio
import numpy as np

# 仿真器规定的 5 个离散动作
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4

# ---------- 和 01 完全相同的"大脑" ----------
def decide(obs):
    front_gap = None
    for v in obs[1:]:
        if v[0] < 0.5:
            continue
        dx, dy = v[1], v[2]
        if abs(dy) < 2 and dx > 0 and (front_gap is None or dx < front_gap):
            front_gap = dx

    if front_gap is None or front_gap > 45:
        return FASTER, "巡航加速"

    if front_gap < 18:
        return SLOWER, "前车过近, 先减速"

    def lane_free(target_dy):
        for v in obs[1:]:
            if v[0] < 0.5:
                continue
            dx, dy = v[1], v[2]
            if abs(dy - target_dy) < 2 and -12 < dx < 30:
                return False
        return True

    if lane_free(-4):
        return LANE_LEFT, "向左变道"
    if lane_free(+4):
        return LANE_RIGHT, "向右变道"
    return SLOWER, "减速跟车"


def main():
    env = gym.make("highway-v0", render_mode="rgb_array")  # 不弹窗, 返回画面数组
    env.unwrapped.configure({
        # 眼睛: 与 01 相同的米制相对坐标
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": False,
            "see_behind": True,
        },
        # 节奏: 仿真和决策都 15Hz → 每秒15帧, 视频流畅
        "simulation_frequency": 15,
        "policy_frequency": 15,
        # 画面: 比默认更大更清楚
        "screen_width": 900,
        "screen_height": 220,
        "scaling": 7,
        "centering_position": [0.3, 0.5],
    })

    os.makedirs("runs", exist_ok=True)
    obs, info = env.reset(seed=42)

    frames, total_reward, crashes, reasons = [], 0.0, 0, {}
    STEPS = 450  # 450帧 / 15fps = 30秒
    for step in range(STEPS):
        action, reason = decide(obs)
        reasons[reason] = reasons.get(reason, 0) + 1
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(np.array(env.render()))   # 抓下这一帧画面
        total_reward += reward
        if info.get("crashed", False):
            crashes += 1
        if terminated or truncated:
            obs, info = env.reset()
        if step % 75 == 0:
            print(f"录制中... {step}/{STEPS} 帧")

    imageio.mimwrite("runs/highway_demo.mp4", frames, fps=15)
    env.close()

    print(f"\n=== 录制完成: runs/highway_demo.mp4 ({STEPS/15:.0f}秒) ===")
    print(f"碰撞 {crashes} 次 | 总奖励 {total_reward:.1f}")
    print("决策分布:", {k: v for k, v in sorted(reasons.items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
