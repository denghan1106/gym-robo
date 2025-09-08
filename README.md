# Gymnasium Robotics - Shadow Hand PPO Training

这是一个基于强化学习的机械手抓取训练项目，使用 PPO（Proximal Policy Optimization）算法训练 Shadow Dexterous Hand 在 MuJoCo 仿真环境中进行精确的物体抓取任务。

## 🎯 项目目标

训练一个智能机械手系统，使其能够：
- 在指定高度（0.5m）保持待命状态
- 精确下降到低位（0.02m）抓取地面上的方块
- 学习自然的抓取姿势和稳定的物体操作

## 🏗️ 项目结构

```
├── train_ppo.py           # 自定义PPO训练主程序
├── train_sb3.py           # Stable-Baselines3 PPO训练程序
├── ppo.py                 # 自定义PPO算法实现
├── network.py             # Actor-Critic神经网络架构
├── reward_utils.py        # 自定义奖励函数
├── reward_lift_sb3.py     # SB3版本奖励函数
├── checkpoint_utils.py    # 模型保存/加载工具
├── visualize_local.py     # 训练结果可视化
├── visualize_sb3.py       # SB3模型可视化
├── checkpoints/           # 训练检查点目录
├── rllib_checkpoints/     # RLLib检查点目录
├── gymnasium_robotics/    # 机器人环境包
└── tests/                 # 测试文件
```

## ✨ 核心特性

### 🤖 智能机械手控制
- **Shadow Dexterous Hand**: 22自由度高精度机械手
- **触觉反馈**: 92个触觉传感器提供丰富的接触信息
- **精确定位**: 3D空间中的精确手臂位置控制

### 🧠 强化学习算法
- **自定义PPO**: 从零实现的PPO算法，针对机械手任务优化
- **Stable-Baselines3**: 使用成熟的RL库进行对比训练
- **Actor-Critic架构**: 双网络结构，策略网络和价值网络并行训练

### 🎁 智能奖励系统
- **多层次奖励**: 接触、抓取、举升的阶段性奖励
- **自然抓取**: 鼓励大拇指配合其他手指的自然抓取姿势
- **稳定性奖励**: 奖励多点接触和稳定夹持
- **安全约束**: 穿模惩罚和滑落检测

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
pip install gymnasium-robotics
pip install mujoco>=2.2.0
pip install torch
pip install stable-baselines3  # 可选，用于SB3版本
pip install numpy>=1.21.0
```

### 安装项目

```bash
git clone https://github.com/denghan1106/gym-robo.git
cd gym-robo
pip install -e .
```

### 开始训练

#### 方法1: 自定义PPO训练
```bash
python train_ppo.py
```

#### 方法2: Stable-Baselines3训练
```bash
python train_sb3.py
```

### 可视化训练结果

```bash
# 可视化自定义PPO结果
python visualize_local.py

# 可视化SB3结果
python visualize_sb3.py
```

## 📊 训练配置

### 默认超参数
- **训练回合数**: 300 episodes
- **每回合最大步数**: 200-300 steps
- **学习率**: 3e-4
- **折扣因子**: 0.99
- **PPO截断参数**: 0.2
- **更新频率**: 每回合结束后更新

### 动作空间
- **手指控制**: 22维连续动作空间，控制各个关节角度
- **手臂控制**: 3维位置控制 (x, y, z)
- **动作范围**: [-1, 1] 标准化输入，自动映射到物理控制范围

### 状态空间
- **机械手状态**: 关节角度、角速度
- **物体状态**: 位置、姿态、速度
- **触觉信息**: 92维触觉传感器数据
- **目标信息**: 期望的物体位置

## 🎯 核心算法

### 动作缩放机制
```python
def scale_action(raw_a):
    # 手臂位置控制
    a[X_IDX] = raw_a[X_IDX] * 0.03  # X轴微调
    a[Y_IDX] = raw_a[Y_IDX] * 0.03  # Y轴微调
    a[Z_IDX] = (raw_a[Z_IDX] + 1.0) * 0.5 * (1.8 - 0.02) + 0.02  # Z轴全范围
    
    # 手指控制：不同手指采用不同策略
    # 食指：较小弯曲，中指：中等弯曲，等等
```

### 奖励函数设计
```python
def compute_custom_reward(obs, achieved_goal, desired_goal, touch_values):
    # 1. 基础接触奖励
    # 2. 抓取稳定性奖励
    # 3. 举升高度奖励
    # 4. 自然抓取姿势奖励
    # 5. 安全约束惩罚
```

## 📈 训练进度监控

训练过程中会实时输出以下信息：
- 每回合总奖励
- 抓取成功率 (Y/N)
- 举升成功率 (Y/N)
- 回合步数

示例输出：
```
EP 0150/0300 | steps 200 | R   45.23 | grasp Y | lift Y
EP 0151/0300 | steps 180 | R   52.10 | grasp Y | lift N
```

## 💾 模型保存与加载

### 自动保存
- 每100回合自动保存检查点
- 训练结束保存最终模型
- 保存位置：`checkpoints/default_env_ppo/` 或 `checkpoints/sb3_ppo_simple/`

### 手动加载
```python
from checkpoint_utils import load_model
load_model(actor, critic, "checkpoints/default_env_ppo", "ppo_episode_300_final")
```

## 🔧 自定义配置

### 修改训练参数
在 `train_ppo.py` 或 `train_sb3.py` 中修改：
```python
TRAINING_EPISODES = 500      # 增加训练回合数
MAX_STEPS_PER_EPISODE = 250  # 调整每回合步数
Z_INIT = 0.8                 # 修改初始高度
```

### 自定义奖励函数
在 `reward_utils.py` 中修改 `compute_custom_reward()` 函数来调整奖励机制。



## 🙏 致谢

- [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) - 提供机器人环境
- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法库

