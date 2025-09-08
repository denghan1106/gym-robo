'''
import os
from gymnasium.envs.registration import register, registry

# 导入原始模块，方便修改其常量
from gymnasium_robotics.envs.shadow_dexterous_hand import manipulate_block as mb

_THIS_DIR = os.path.dirname(__file__)
_XML_PATH = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "assets", "hand", "manipulate_block_custom.xml")
)

# ① 动态替换模块里的常量
mb.MODEL_XML_PATH = _XML_PATH

# ② 自己什么都不用改写，直接复用原类
BaseEnvClass = mb.MujocoHandBlockEnv   # 若你用 mujoco-py 则换成 MujocoPyHandBlockEnv


def register_custom_shadow_env():
    if "ShadowHandGraspEllipsoid-v0" in registry:
        return

    register(
        id="ShadowHandGraspEllipsoid-v0",
        entry_point=(
            "gymnasium_robotics.envs.shadow_dexterous_hand.register_envs:BaseEnvClass"
        ),
        max_episode_steps=200,
    )
'''
'''
import os
from gymnasium_robotics.envs.shadow_dexterous_hand import manipulate_block as mb

_XML_PATH = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "assets", "hand", "manipulate_block_custom.xml")
)
mb.MANIPULATE_BLOCK_XML = _XML_PATH   # 只做这一件事
'''