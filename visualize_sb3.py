#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Stable-Baselines3 训练好的模型进行可视化
"""

import os
import numpy as np
import mujoco
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors import (
    MujocoHandBlockTouchSensorsEnv,
)
from stable_baselines3 import PPO
from reward_lift_sb3 import compute_custom_rewardsb3

# ---------- 常量 ----------
Z_MIN, Z_MAX = 0.02, 1.8          # arm:z 的物理控制范围
Z_INIT       = 0.5                # 每次 reset 时的起始高度
Z_IDX        = -1                 # arm:z 在动作向量中的索引（最后一位）
X_IDX        = -3                 # arm:x 在动作向量中的索引
Y_IDX        = -2                 # arm:y 在动作向量中的索引

# ---------- XML 路径 ----------
BaseEnv = MujocoHandBlockTouchSensorsEnv
BaseEnv.MANIPULATE_BLOCK_XML_PATH = os.path.join("hand", "manipulate_block_touch_sensor.xml")

# ---------- 动作缩放 ----------
def scale_action(raw_a: np.ndarray) -> np.ndarray:
    """
    将 tanh 输出的 raw_a ∈ [-1,1] 映射到物理控制范围。
    - arm:x, arm:y (倒数第3、2位) 映射到位置控制范围
    - arm:z (最后一位) 映射到速度控制范围
    - 手指关节映射到各自的物理范围
    """
    a = raw_a.copy()
    
    # 处理 arm:x (倒数第3位) - 位置控制
    # 将 [-1, 1] 映射到位置范围 [-0.01, 0.01]
    a[X_IDX] = raw_a[X_IDX] * 0.03
    
    # 处理 arm:y (倒数第2位) - 位置控制
    # 将 [-1, 1] 映射到位置范围 [-0.01, 0.01]
    a[Y_IDX] = raw_a[Y_IDX] * 0.03
    
    # 处理 arm:z (最后一位) - 速度控制
    # 将 [-1, 1] 映射到速度范围 [-2.0, 2.0] m/s
    a[Z_IDX] = (raw_a[Z_IDX] + 1.0) * 0.5 * (1.8 - 0.02) + 0.02
    
    # 处理手指关节 (除了最后3位的所有维度)
    for i in range(len(raw_a) - 3):
        if i < 4:  # 食指关节 (0-3)
            a[i] = 0.15  # 较小的伸直力，食指稍微弯曲
        elif i < 8:  # 中指关节 (4-7)
            a[i] = 0.2  # 中等伸直力
        elif i < 12:  # 无名指关节 (8-11)
            a[i] = 0.25  # 较大伸直力
        elif i < 17:  # 小指关节 (12-16)
            a[i] = 0.3  # 最大伸直力，小指最直
        else:  # 大拇指关节 (17-21)
            a[i] = 0.1  # 大拇指保持弯曲状态，便于抓取
    
    return a

# ---------- 自定义环境 ----------
OPEN_POSE = {
    "robot0:THJ1": -0.20, "robot0:THJ2":  0.20,
    "robot0:FFJ1":  0.00, "robot0:FFJ2":  0.05,
    "robot0:MFJ1":  0.00, "robot0:MFJ2":  0.05,
    "robot0:RFJ1":  0.00, "robot0:RFJ2":  0.05,
    "robot0:LFJ1":  0.00, "robot0:LFJ2":  0.05,
}

class GroundSpawnEnv(BaseEnv):
    def _reset_sim(self):
        if not super()._reset_sim():
            return False
        
        # 把方块放地面
        obj_qadr = self.model.joint("object:joint").qposadr[0]
        self.data.qpos[obj_qadr + 2] = 0.025

        # arm:z ctrl 初始化到 1.2
        a_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_arm_z"
        )
        self.data.ctrl[a_id] = Z_INIT

        # 手指张开
        for jname, q_open in OPEN_POSE.items():
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            if jid == -1:
                continue
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = q_open

        mujoco.mj_forward(self.model, self.data)
        return True

# ---------- 包装环境 ----------
class SB3Wrapper:
    def __init__(self, env):
        self.env = env
        self.prev_z = None
        self.prev_hand_z = None
        
    def reset(self):
        obs_dict, _ = self.env.reset()
        obs = obs_dict["observation"]
        self.prev_z = obs_dict["achieved_goal"][2]
        self.prev_hand_z = obs[2]  # 手的初始高度
        return obs, {}
    
    def step(self, action):
        scaled_action = scale_action(action)
        obs_dict, _, term, trunc, _ = self.env.step(scaled_action)
        done = term or trunc
        
        obs = obs_dict["observation"]
        achieved = obs_dict["achieved_goal"]
        desired = obs_dict["desired_goal"]
        touch = obs[-92:]
        
        reward, _ = compute_custom_rewardsb3(obs, achieved, desired, touch, self.prev_z, self.prev_hand_z, height_threshold=0.07)
        self.prev_z = achieved[2]
        self.prev_hand_z = obs[2]  # 更新手的前一高度
        
        return obs, reward, done, False, {}
    
    @property
    def observation_space(self):
        return self.env.observation_space["observation"]
    
    @property
    def action_space(self):
        return self.env.action_space

# ---------- 主程序 ----------
def main():
    print("创建可视化环境 …")
    
    # 创建环境
    env = GroundSpawnEnv(render_mode="human")  # 启用渲染
    env = TimeLimit(env, max_episode_steps=200)
    env = SB3Wrapper(env)
    
    print("环境就绪。")
    
    # 加载模型
    model_path = "checkpoints/sb3_ppo_simple/sb3_ppo_simple_episode_300_final"
    if not os.path.exists(model_path + ".zip"):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train_sb3.py 训练模型")
        return
    
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)
    print("模型加载完成。")
    
    # 开始可视化
    print("\n========== 开始可视化 ==========")
    
    for ep in range(10):
        print(f"\n--- Episode {ep+1}/10 ---")
        
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        print(f"Episode finished in {step_count} steps. Total reward: {total_reward:.2f}")
    
    print("\n========== 可视化完成 ==========")
    env.env.close()

if __name__ == "__main__":
    main() 