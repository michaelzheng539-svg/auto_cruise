"""
01_hello_highway.py — 你的第一个自动驾驶程序
==============================================
它在一个叫 highway-env 的仿真环境里开车。整个程序就是一个循环：

    眼睛(观测) → 大脑(决策) → 手脚(动作) → 仿真器算出下一帧 → 再回到眼睛...

特斯拉等所有自动驾驶系统，最底层都是这个骨架；
区别只是"大脑"从我们这种手写规则，换成了训练出来的神经网络。

附一个真实教训（第一版 bug）：我们起初以为观测是"米制相对距离"，
但这个版本默认输出"归一化数值"——大脑理解错了眼睛看到的数字，400 帧撞了 41 次。
所以下面显式配置了观测格式。自动驾驶第一课：先弄清传感器数据的真实含义。

运行方式:
    .venv\\Scripts\\python scripts\\01_hello_highway.py
"""
import time

import gymnasium as gym
import highway_env  # noqa: F401  (import 即注册仿真环境)

# 仿真器规定的 5 个离散动作（离散 = 只能从这几个选项里选）
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


def decide(obs):
    """手写策略: 前方空旷就加速巡航；有车就变道躲避；躲不开就刹车。

    obs 是一个 8x5 的表格: 每行一辆车，5 列 =
    [是否存在, 前方距离x, 横向偏移y, 前向速度vx, 横向速度vy]
    第 0 行是自己（注意：自己这行的 x/y 是绝对坐标，库的怪癖）；
    其余行都是"以自己为原点"的相对距离，单位米。
    """
    # --- 第1步: 找同车道正前方最近的车 (车道宽4米, |横向偏移|<2米算同一车道) ---
    front_gap = None
    for v in obs[1:]:
        if v[0] < 0.5:          # presence<0.5 → 这个位置没有车
            continue
        dx, dy = v[1], v[2]     # 前方距离 / 横向偏移(左负右正), 单位米
        if abs(dy) < 2 and dx > 0 and (front_gap is None or dx < front_gap):
            front_gap = dx

    # --- 第2步: 前方空旷(或超过45米) → 巡航加速 ---
    if front_gap is None or front_gap > 45:
        return FASTER, "前方空旷, 巡航加速"

    # --- 第3步: 前车太近(<18米) → 先刹车保命, 变道的事下一帧再说 ---
    if front_gap < 18:
        return SLOWER, f"前车过近({front_gap:.0f}米) → 先减速"

    # --- 第4步: 前方有车 → 检查旁边车道有没有空位 ---
    def lane_free(target_dy):
        """target_dy=-4 是左邻车道, +4 是右邻车道"""
        for v in obs[1:]:
            if v[0] < 0.5:
                continue
            dx, dy = v[1], v[2]
            if abs(dy - target_dy) < 2 and -12 < dx < 30:  # 目标车道前后有车占位
                return False
        return True

    if lane_free(-4):
        return LANE_LEFT, f"前方{front_gap:.0f}米有车 → 向左变道"
    if lane_free(+4):
        return LANE_RIGHT, f"前方{front_gap:.0f}米有车 → 向右变道"

    # --- 第4步: 两边都堵 → 刹车跟车 ---
    return SLOWER, f"前方{front_gap:.0f}米有车, 两侧被占 → 减速"


def main():
    env = gym.make("highway-v0", render_mode="human")

    # 显式配置"眼睛"的数据格式（吃过的亏: 默认格式是归一化数值, 不是米）
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,               # 能看到 7 辆周围车辆
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,                 # 相对坐标
            "normalize": False,                # 不归一化, 直接用米
            "see_behind": True,                # 也看得到身后的车(变道判断要用)
        },
    })

    # seed=42: 固定随机器的种子 → 每次运行遇到的车流一模一样(可复现实验的第一步)
    obs, info = env.reset(seed=42)
    print("=== 初始观测 obs (8辆车 x 5个数值) ===")
    print(obs)
    print("列含义: [是否存在, 前方距离x, 横向偏移y, 前向速度vx, 横向速度vy]\n")

    total_reward, crashes, steps = 0.0, 0, 0
    for step in range(400):  # 最多开400帧
        action, reason = decide(obs)                 # ① 大脑决策
        obs, reward, terminated, truncated, info = env.step(action)  # ② 交给仿真器
        env.render()                                 # ③ 画出一帧画面
        time.sleep(0.05)                             # 放慢到约20帧/秒, 方便肉眼观察

        total_reward += reward
        steps += 1
        if step % 10 == 0:                           # 每10帧汇报一次大脑的想法
            print(f"第{step:3d}帧 | {reason} | 累计奖励 {total_reward:6.1f}")

        if info.get("crashed", False):
            crashes += 1
        if terminated or truncated:                  # 撞车/出局 → 重新开局
            print("—— 本局结束, 重新开局 ——")
            obs, info = env.reset()

    print(f"\n=== 汇总: 行驶{steps}帧, 碰撞{crashes}次, 总奖励{total_reward:.1f} ===")
    env.close()


if __name__ == "__main__":
    main()
