#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载训练好的权重，实时渲染手臂抓块
"""

import os, time, torch, mujoco
import numpy as np
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors import MujocoHandBlockTouchSensorsEnv

from train_ppo import GroundSpawnEnv, scale_action, Z_IDX, X_IDX, Y_IDX  # 直接复用函数/常量
from ppo import PPO
from checkpoint_utils import load_model

# ---------- 可视化配置 ----------
CHECKPOINT_DIR  = "checkpoints/default_env_ppo"
MODEL_FILENAME  = "ppo_episode_300_final"   # 改成你的最终文件名
NUM_EPISODES    = 15
MAX_STEPS       = 100
WIN_W, WIN_H    = 1280, 720
# --------------------------------

BaseEnv = MujocoHandBlockTouchSensorsEnv
BaseEnv.MANIPULATE_BLOCK_XML_PATH = os.path.join("hand", "manipulate_egg_touch_sensors.xml")

def main():
    print("创建可视化环境 …")
    env = GroundSpawnEnv(render_mode="human", width=WIN_W, height=WIN_H)
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    print("环境 OK。")

    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.shape[0]
    ppo = PPO(obs_dim, act_dim)

    load_model(ppo.actor, ppo.critic, CHECKPOINT_DIR, MODEL_FILENAME)
    ppo.actor.eval();  ppo.critic.eval()
    print("模型已加载。")

    print("\n========== 开始可视化 ==========")
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES} ---")
        obs_dict, _ = env.reset()
        obs = obs_dict["observation"]
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                raw_a, _ = ppo.actor(torch.FloatTensor(obs).unsqueeze(0))
                raw_a = raw_a.squeeze().cpu().numpy()
            env_a = scale_action(raw_a)

            _, _, term, trunc, _ = env.step(env_a)
            done = term or trunc
            obs   = env.unwrapped._get_obs()["observation"]
            steps += 1
            time.sleep(0.02)

        print(f"Episode finished in {steps} steps.")

    env.close()
    print("\n========== 结束 ==========")

if __name__ == "__main__":
    main()