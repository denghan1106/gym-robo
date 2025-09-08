#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO 训练（SB3 版本）：让 hand_base 在 0.5 m 待命，并能下压到 0.02 m 抓取方块
"""

import os
import numpy as np
import mujoco
from gymnasium.wrappers import TimeLimit
from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors import (
    MujocoHandBlockTouchSensorsEnv,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from reward_lift_sb3 import compute_custom_rewardsb3

# ---------- 常量 ----------
Z_MIN, Z_MAX = 0.02, 1.8          # arm:z 的物理控制范围
Z_INIT       = 0.5                # 每次 reset 时的起始高度
Z_IDX        = -1                 # arm:z 在动作向量中的索引（最后一位）
X_IDX        = -3                 # arm:x 在动作向量中的索引
Y_IDX        = -2                 # arm:y 在动作向量中的索引

TRAINING_EPISODES     = 300        # 更多训练轮数
MAX_STEPS_PER_EPISODE = 300        # 更多步数
SAVE_FREQUENCY        = 100
CHECKPOINT_DIR        = "checkpoints/sb3_ppo_simple"
N_ENVS                = 8          # 更多并行环境

# ---------- XML 路径 ----------
BaseEnv = MujocoHandBlockTouchSensorsEnv
BaseEnv.MANIPULATE_BLOCK_XML_PATH = os.path.join("hand", "manipulate_block_touch_sensors.xml")

# ---------- 简化的动作缩放 ----------
def scale_action_simple(raw_a: np.ndarray) -> np.ndarray:
    """
    简化的动作缩放：基于成功的原始脚本
    """
    a = raw_a.copy()
    
    # 手臂控制 (最后3位)
    a[-3] = raw_a[-3] * 0.03  # x位置
    a[-2] = raw_a[-2] * 0.03  # y位置  
    a[-1] = (raw_a[-1] + 1.0) * 0.5 * (1.8 - 0.02) + 0.02  # z位置
    
    # 手指控制：使用固定的抓取姿势，与原始成功脚本一致
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

# ---------- 初始姿势 ----------
OPEN_POSE = {
    "robot0:THJ1": -0.20, "robot0:THJ2":  0.20,
    "robot0:FFJ1":  0.00, "robot0:FFJ2":  0.05,
    "robot0:MFJ1":  0.00, "robot0:MFJ2":  0.05,
    "robot0:RFJ1":  0.00, "robot0:RFJ2":  0.05,
    "robot0:LFJ1":  0.00, "robot0:LFJ2":  0.05,
}

class SimpleEnv(BaseEnv):
    def _reset_sim(self):
        if not super()._reset_sim():
            return False
        
        # 设置方块位置
        obj_qadr = self.model.joint("object:joint").qposadr[0]
        self.data.qpos[obj_qadr + 2] = 0.025
        
        # 设置手臂初始位置
        a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_arm_z")
        self.data.ctrl[a_id] = Z_INIT
        
        # 设置手指姿势
        for jname, q_open in OPEN_POSE.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                qadr = self.model.jnt_qposadr[jid]
                self.data.qpos[qadr] = q_open
        
        mujoco.mj_forward(self.model, self.data)
        return True

# ---------- SB3 包装 ----------
import gymnasium as gym
class SimpleWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.prev_z = None
        self.prev_hand_z = None
        self.observation_space = env.observation_space["observation"]
        self.action_space = env.action_space
        self.episode_grasped = False
        self.episode_lifted = False
    
    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        obs = obs_dict["observation"]
        self.prev_z = obs_dict["achieved_goal"][2]
        self.prev_hand_z = obs[2]  # 手的初始高度
        self.episode_grasped = False
        self.episode_lifted = False
        return obs, {}
    
    def step(self, action):
        scaled_action = scale_action_simple(action)
        obs_dict, _, term, trunc, _ = self.env.step(scaled_action)
        done = term or trunc
        obs = obs_dict["observation"]
        achieved = obs_dict["achieved_goal"]
        desired = obs_dict["desired_goal"]
        touch = obs[-92:]
        reward, r_info = compute_custom_rewardsb3(obs, achieved, desired, touch, self.prev_z, self.prev_hand_z, height_threshold=0.07)
        self.prev_z = achieved[2]
        self.prev_hand_z = obs[2]  # 更新手的前一高度
        
        # 跟踪episode级别的grasp和lift状态
        if r_info["grasped"]:
            self.episode_grasped = True
        if r_info["lifted"]:
            self.episode_lifted = True
            
        return obs, reward, done, False, {
            "r_info": r_info,
            "episode_grasped": self.episode_grasped,
            "episode_lifted": self.episode_lifted
        }

# ---------- 主程序 ----------
def make_env():
    def _thunk():
        env = SimpleEnv(render_mode=None)
        env = TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
        return SimpleWrapper(env)
    return _thunk

def main():
    print("创建简化训练环境...")
    vec_env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
    print("环境就绪。")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQUENCY * MAX_STEPS_PER_EPISODE,
        save_path=CHECKPOINT_DIR,
        name_prefix="sb3_ppo_simple"
    )
    
    simple_callback = SimpleCallback()
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,  # 更高的学习率
        n_steps=MAX_STEPS_PER_EPISODE,
        batch_size=128,  # 更大的batch size
        n_epochs=4,  # 减少epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.05,  # 更高的熵系数
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        tensorboard_log=f"{CHECKPOINT_DIR}/tensorboard_logs",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 更小的网络
            log_std_init=2.0,  # 更高的初始探索
        )
    )
    
    print("PPO 模型创建完成。\n========== 开始训练 ==========")
    model.learn(
        total_timesteps=TRAINING_EPISODES * MAX_STEPS_PER_EPISODE,
        callback=[checkpoint_callback, simple_callback],
        progress_bar=True
    )
    print("\n========== 训练完成 ==========")
    model.save(os.path.join(CHECKPOINT_DIR, f"sb3_ppo_simple_episode_{TRAINING_EPISODES}_final"))
    vec_env.close()

# ---------- 自定义回调函数 ----------
class SimpleCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0
        
    def _on_rollout_end(self) -> None:
        """在每个rollout结束时调用"""
        self.rollout_count += 1
        print(f"Rollout {self.rollout_count} completed")
        
    def _on_step(self) -> bool:
        return True

if __name__ == "__main__":
    main() 