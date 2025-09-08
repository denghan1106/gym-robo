#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO 训练：让 hand_base 在 1.2 m 待命，并能下压到 0.02 m 抓取方块
"""

import os
import numpy as np
import torch
import mujoco
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors import (
    MujocoHandEggTouchSensorsEnv,
)
from ppo import PPO
from checkpoint_utils import save_model
from reward_utils import compute_custom_reward

# ---------- 常量 ----------
Z_MIN, Z_MAX = 0.02, 1.8
Z_INIT       = 0.5
Z_IDX        = -1
X_IDX        = -3
Y_IDX        = -2
TRAINING_EPISODES     = 300
MAX_STEPS_PER_EPISODE = 200
SAVE_FREQUENCY        = 100
CHECKPOINT_DIR        = "checkpoints/default_env_ppo"

# ---------- XML 路径 ----------
BaseEnv = MujocoHandEggTouchSensorsEnv
BaseEnv.MANIPULATE_BLOCK_XML_PATH = os.path.join("hand", "manipulate_egg_touch_sensors.xml")

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
    
    # 处理 arm:z (最后一位) - 位置控制
    # 将 [-1, 1] 映射到位置范围 [0.02, 1.8]
    a[Z_IDX] = (raw_a[Z_IDX] + 1.0) * 0.5 * (1.8 - 0.02) + 0.02
    
    # 处理手指关节 (除了最后3位的所有维度)
    # 假设手指关节的物理范围是 [0, 1.571] (对应 range="0 1.571")
    for i in range(len(raw_a) - 3):  # 除了最后3位
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

# ---------- OPEN_POSE & 环境 ----------
OPEN_POSE = {
    "robot0:THJ1": -0.20, "robot0:THJ2":  0.20,
    "robot0:FFJ1":  0.00, "robot0:FFJ2":  0.05,
    "robot0:MFJ1":  0.00, "robot0:MFJ2":  0.05,
    "robot0:RFJ1":  0.00, "robot0:RFJ2":  0.05,
    "robot0:LFJ1":  0.00, "robot0:LFJ2":  0.05,
}
class GroundSpawnEnv(BaseEnv):
    def _reset_sim(self):
        # ---------- 1. 先执行父类复位 ----------
        if not super()._reset_sim():
            return False
        # 放到 _reset_sim 里 super()._reset_sim() 之后，或者在可视化脚本里
        
        # ---------- 2. 把方块放地面 ----------
        obj_qadr = self.model.joint("object:joint").qposadr[0]
        self.data.qpos[obj_qadr + 2] = 0.025   # 只改 z

        # ---------- 3. arm:z ctrl 初始化到 1.2 ----------
        a_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_arm_z"
        )
        self.data.ctrl[a_id] = Z_INIT          # Z_INIT = 1.2

        # ---------- 4. 手指张开 ----------
        for jname, q_open in OPEN_POSE.items():
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            if jid == -1:
                print(f"[WARN] 找不到关节 {jname}，已跳过")
                continue
            qadr = self.model.jnt_qposadr[jid]  # 该关节在 qpos 中的索引
            self.data.qpos[qadr] = q_open       # 设定张开角度

        # ---------- 5. 推进一次前向动力学 ----------
        mujoco.mj_forward(self.model, self.data)
        return True

# ---------- 主程序 ----------
def main():
    print("创建环境 …")
    env = GroundSpawnEnv(render_mode=None)
    env = TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
    print("环境就绪。")
    
    # 打印调试信息
    print(f"动作空间: {env.action_space}")
    print(f"动作空间形状: {env.action_space.shape}")
    print(f"动作空间范围: low={env.action_space.low}, high={env.action_space.high}")
    print(f"观察空间: {env.observation_space['observation'].shape}")

    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.shape[0]           # 已包含 arm:z
    ppo = PPO(obs_dim, act_dim, lr=3e-4, K_epochs=10)
    print(f"PPO 初始化完成：obs_dim={obs_dim}, act_dim={act_dim}")

    print("\n========== 开始训练 ==========")
    for ep in range(TRAINING_EPISODES):
        obs_dict, _ = env.reset()
        obs = obs_dict["observation"]
        prev_z = obs_dict["achieved_goal"][2]

        done = False
        total_reward = 0.0
        step_cnt = 0
        ep_lifted = ep_grasped = False

        while not done:
            raw_a, logp = ppo.select_action(obs)      # raw_a ∈ [-1,1]
            env_a      = scale_action(raw_a)          # 映射到物理范围

            next_obs_dict, _, term, trunc, _ = env.step(env_a)
            done = term or trunc

            next_obs = next_obs_dict["observation"]
            achieved = next_obs_dict["achieved_goal"]
            desired  = next_obs_dict["desired_goal"]
            touch    = next_obs[-92:]
            # ...existing code...

            reward, r_info = compute_custom_reward(obs, achieved, desired, touch, prev_z, 
                                                height_threshold=0.07)
            ppo.store_transition(obs, raw_a, reward, done, logp, next_obs)

            obs      = next_obs
            prev_z   = achieved[2]
            total_reward += reward
            step_cnt += 1
            if r_info["lifted"]:  ep_lifted  = True
            if r_info["grasped"]: ep_grasped = True

        ppo.update()
        print(f"EP {ep+1:04d}/{TRAINING_EPISODES} | steps {step_cnt:03d} | "
              f"R {total_reward:8.2f} | grasp {'Y' if ep_grasped else 'N'} "
              f"| lift {'Y' if ep_lifted else 'N'}"
              )

        if (ep + 1) % SAVE_FREQUENCY == 0:
            save_model(ppo.actor, ppo.critic, act_dim, obs_dim,
                       CHECKPOINT_DIR, f"ppo_episode_{ep+1}")

    print("\n========== 训练完成 ==========")
    save_model(ppo.actor, ppo.critic, act_dim, obs_dim,
               CHECKPOINT_DIR, f"ppo_episode_{TRAINING_EPISODES}_final")
    env.close()

if __name__ == "__main__":
    main()