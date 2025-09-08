import numpy as np

def compute_custom_rewardsb3(obs, achieved_goal, desired_goal, touch_values=None, prev_z=None, prev_hand_z=None, height_threshold=0.37):
    """
    优化版奖励函数：
    - 平衡的奖励设计，避免极端负值
    - 渐进式奖励，鼓励逐步改进
    - 更强的抬升激励
    """
    reward = 0.0
    z_pos = achieved_goal[2]
    hand_z = obs[2]
    touched = False
    grasped = False
    lifted = False
    natural_grasp = False
    penetration_penalty = 0.0
    stable_grasp = False
    multi_contact = False
    fingertip_contact = False
    drop_penalty = 0.0
    height_maintenance_reward = 0.0

    if touch_values is not None:
        touched = np.any(touch_values > 0.0)
        num_touched = np.sum(touch_values > 0.0)
        grasped = num_touched > 8  # 降低抓取阈值
        natural_grasp = _check_natural_grasp(touch_values)
        penetration_penalty = _check_penetration(touch_values) * 0.1  # 进一步减小惩罚
        multi_contact = _check_multi_contact(touch_values)
        fingertip_contact = _check_fingertip_contact(touch_values)
        stable_grasp = _check_stable_grasp(touch_values, grasped)

    drop_penalty = _check_drop_penalty(z_pos, grasped, prev_z) * 0.1  # 进一步减小惩罚
    height_maintenance_reward = _compute_height_maintenance_reward(z_pos, prev_z, grasped) * 3.0  # 增加正向奖励

    # 基础奖励 - 更温和的设计
    reward += _compute_basic_rewards(touched, grasped, z_pos, hand_z, prev_z, height_threshold) * 1.5

    # 新奖励项 - 更平衡的设计
    if natural_grasp:
        reward += 5.0  # 自然抓取姿势奖励
    if multi_contact:
        reward += 4.0  # 多点接触奖励
    if fingertip_contact:
        reward += 3.0  # 指尖传感器奖励
    if stable_grasp:
        reward += 6.0  # 稳定夹持奖励
    
    # 惩罚项
    reward += penetration_penalty
    reward += drop_penalty
    reward += height_maintenance_reward

    # 抬升高度奖励 - 更强的激励
    if z_pos > 0.15:  # 只要离开地面就给奖励
        reward += 10.0 * z_pos  # 基础高度奖励
    if prev_z is not None and z_pos > prev_z:
            reward += 20.0 * (z_pos - prev_z)  # 方块高度增加奖励
    
    # 手高度增加奖励 - 鼓励手向上移动
    if prev_hand_z is not None and hand_z > prev_hand_z:
        hand_height_gain = hand_z - prev_hand_z
        if hand_height_gain > 0.01:  # 手高度显著增加
            reward += 15.0 * hand_height_gain  # 手高度增加奖励
            if grasped:  # 如果同时抓住物体，给予额外奖励
                reward += 25.0 * hand_height_gain  # 抓住物体时手上升的额外奖励
    
    # 目标高度奖励
    if z_pos > height_threshold:
        reward += 50.0  # 达到目标高度的额外奖励

    lifted = z_pos > height_threshold
    info = {
        "touched": touched,
        "grasped": grasped,
        "lifted": lifted,
        "natural_grasp": natural_grasp,
        "penetration_penalty": penetration_penalty,
        "multi_contact": multi_contact,
        "fingertip_contact": fingertip_contact,
        "stable_grasp": stable_grasp,
        "drop_penalty": drop_penalty,
        "height_maintenance_reward": height_maintenance_reward,
        "z_pos": z_pos,
        "reward": reward
    }
    return reward, info

def _check_natural_grasp(touch_values):
    """
    检查是否为自然抓取姿势：
    当大拇指与其他手指均有接触时，判定为自然的夹持动作
    """
    if touch_values is None or len(touch_values) < 99:
        return False
    
    # 大拇指传感器索引（根据XML文件，大拇指传感器从第85个开始）
    # thproximal: 85-89, thmiddle: 90-94, thtip: 95-99
    thumb_sensors = touch_values[84:99]  # 大拇指的15个传感器
    
    # 其他手指传感器（排除大拇指和手掌）
    # 手掌: 0-7, 其他手指: 8-84
    other_finger_sensors = np.concatenate([
        touch_values[8:84]  # 其他手指传感器
    ])
    
    # 检查大拇指是否有接触
    thumb_touching = np.any(thumb_sensors > 0.0)
    
    # 检查其他手指是否有接触
    other_fingers_touching = np.any(other_finger_sensors > 0.0)
    
    # 自然抓取：大拇指与其他手指均有接触
    return thumb_touching and other_fingers_touching

def _check_penetration(touch_values):
    """
    检测穿模惩罚项：
    若任一触觉传感器值异常高，视为"可能穿透物体"
    """
    if touch_values is None:
        return 0.0
    
    # 设置穿模阈值
    penetration_threshold = 8.0  # 提高阈值
    
    # 检查是否有异常高的触觉值
    max_touch_value = np.max(touch_values)
    
    if max_touch_value > penetration_threshold:
        return -2.0  # 减小惩罚
    
    return 0.0

def _check_multi_contact(touch_values):
    """
    检查多点接触：
    鼓励使用多个手指和多个接触点进行抓取
    只考虑指尖附近的传感器，排除远离指尖的接触点
    """
    if touch_values is None:
        return False
    
    # 定义指尖附近的传感器索引（排除手掌和手指根部）
    # 大拇指指尖附近: 95-99 (5个)
    # 食指指尖附近: 20-30 (10个)
    # 中指指尖附近: 45-55 (10个)
    # 无名指指尖附近: 70-80 (10个)
    # 小指指尖附近: 85-94 (10个，但这里是大拇指，所以排除)
    
    fingertip_near_sensors = np.concatenate([
        touch_values[95:99],  # 大拇指指尖
        touch_values[20:30],  # 食指指尖附近
        touch_values[45:55],  # 中指指尖附近
        touch_values[70:80],  # 无名指指尖附近
    ])
    
    # 计算指尖附近的活跃接触点数量
    active_contacts = np.sum(fingertip_near_sensors > 0.05)  # 降低阈值，更容易触发
    
    # 多点接触：至少6个指尖附近的接触点
    return active_contacts >= 6

def _check_fingertip_contact(touch_values):
    """
    检查指尖传感器使用：
    鼓励使用指尖传感器进行精确抓取
    """
    if touch_values is None or len(touch_values) < 99:
        return False
    
    # 指尖传感器索引（根据92个传感器的分布）
    # 大拇指指尖: 95-99 (5个)
    # 其他手指指尖: 约在20-40, 45-65, 70-90范围内
    fingertip_sensors = np.concatenate([
        touch_values[95:99],  # 大拇指指尖
        touch_values[20:25],  # 食指指尖
        touch_values[45:50],  # 中指指尖
        touch_values[70:75],  # 无名指指尖
    ])
    
    # 检查指尖传感器使用数量
    active_fingertips = np.sum(fingertip_sensors > 0.05)  # 降低阈值
    
    # 使用至少2个指尖传感器
    return active_fingertips >= 2

def _check_stable_grasp(touch_values, grasped):
    """
    检查稳定夹持：
    基于接触力的稳定性和分布来判断夹持是否稳定
    """
    if not grasped or touch_values is None:
        return False
    
    # 计算接触力的标准差（稳定性指标）
    active_touches = touch_values[touch_values > 0.05]  # 降低阈值
    if len(active_touches) < 3:  # 降低要求
        return False
    
    # 接触力稳定性：标准差小于1.0（放宽要求）
    touch_std = np.std(active_touches)
    force_stable = touch_std < 1.0
    
    # 接触点分布：至少分布在2个不同区域（降低要求）
    # 简单检查：手掌、大拇指、其他手指都有接触
    palm_contact = np.any(touch_values[0:8] > 0.05)
    thumb_contact = np.any(touch_values[84:99] > 0.05)
    other_finger_contact = np.any(touch_values[8:84] > 0.05)
    
    distribution_good = palm_contact + thumb_contact + other_finger_contact >= 2
    
    return force_stable and distribution_good

def _check_drop_penalty(z_pos, grasped, prev_z):
    """
    检查抓后滑落惩罚：
    如果物体高度突然下降且之前是抓取状态，给予惩罚
    """
    if not grasped or prev_z is None:
        return 0.0
    
    # 如果物体高度下降超过0.03，视为滑落
    height_drop = prev_z - z_pos
    if height_drop > 0.03:
        return -3.0  # 温和的滑落惩罚
    
    # 如果物体高度很低（接近地面），也给予惩罚
    if z_pos < 0.05:
        return -2.0  # 温和的掉落惩罚
    
    return 0.0

def _compute_basic_rewards(touched, grasped, z_pos, hand_z, prev_z, height_threshold):
    """
    计算基础奖励项（平衡设计，避免极端值）
    """
    reward = 0.0
    
    # 1. 接触物体奖励
    if touched:
        reward += 1.0
    
    # 2. 抓紧物体奖励
    if grasped:
        reward += 2.0
    
    # 3. 抓紧后抬升奖励
    lifted = z_pos > height_threshold
    if grasped and lifted:
        reward += 3.0
    
    # 4. 渐进式高度奖励
    target_z = 1.2
    if grasped:
        # 基础高度奖励
        if z_pos > 0.1:  # 离开地面
            reward += 2.0
        if z_pos > 0.3:  # 中等高度
            reward += 2.0
        if z_pos > 0.6:  # 较高高度
            reward += 3.0
        if z_pos > target_z:  # 目标高度
            reward += 5.0
        
        # 高度提升奖励
        if prev_z is not None and z_pos > prev_z:
            height_gain = z_pos - prev_z
            if height_gain > 0.005:  # 有提升
                reward += height_gain * 50  # 适度的提升奖励
    
    # 5. 接近物体奖励（未抓住时）
    if not grasped and prev_z is not None:
        z_change = hand_z - prev_z
        if z_change < -0.005:  # 向下移动
            reward += 0.5
    
    # 6. 掉落惩罚（温和）
    if z_pos < 0.05:
        reward -= 2.0
    
    # 7. 高度下降惩罚（抓住后，温和）
    if grasped and prev_z is not None:
        height_drop = prev_z - z_pos
        if height_drop > 0.01:  # 高度下降
            reward -= height_drop * 10  # 适度的下降惩罚
    
    return reward

def _compute_height_maintenance_reward(z_pos, prev_z, grasped, target_z=1.2):
    """
    专门计算保持高度的奖励
    这个函数可以被其他奖励函数调用
    """
    if not grasped:
        return 0.0
    
    reward = 0.0
    
    # 1. 高度稳定性奖励
    if prev_z is not None:
        height_change = abs(z_pos - prev_z)
        if height_change < 0.01:  # 高度变化很小，说明很稳定
            reward += 3.0
        elif height_change < 0.02:  # 高度变化较小
            reward += 1.0
    
    # 2. 目标高度维持奖励
    height_error = abs(z_pos - target_z)
    if height_error < 0.05:  # 非常接近目标高度
        reward += 5.0
    elif height_error < 0.1:  # 接近目标高度
        reward += 3.0
    elif height_error < 0.2:  # 在合理范围内
        reward += 1.0
    
    # 3. 高度提升奖励
    if prev_z is not None and z_pos > prev_z:
        height_gain = z_pos - prev_z
        if height_gain > 0.005:  # 有提升
            reward += height_gain * 30
    
    # 4. 高度下降惩罚
    if prev_z is not None and z_pos < prev_z:
        height_drop = prev_z - z_pos
        if height_drop > 0.01:  # 有下降
            reward -= height_drop * 25
    
    return reward

def compute_advanced_reward(obs, achieved_goal, desired_goal, touch_values=None, prev_z=None, 
                          height_threshold=0.07, step_count=0, max_steps=100):
    """
    高级奖励函数，包含更多可拓展功能：
    
    新增功能：
    - 时间稳定性奖励（抓住后保持n步）
    - 手指角度评估（避免"反手"或"翻手"抓取）
    - 力传感器稳定性检测
    """
    # 基础奖励
    base_reward, base_info = compute_custom_rewardsb3(obs, achieved_goal, desired_goal, 
                                                  touch_values, prev_z, height_threshold)
    
    reward = base_reward
    
    # 新增奖励项
    if base_info["grasped"]:
        # 时间稳定性奖励（抓住后保持时间越长奖励越高）
        stability_bonus = min(step_count * 0.1, 2.0)  # 最多2.0奖励
        reward += stability_bonus
        
        # 可以在这里添加更多高级功能
        # 例如：手指角度评估、力传感器检测等
    
    # 更新info字典
    advanced_info = base_info.copy()
    advanced_info["stability_bonus"] = stability_bonus if base_info["grasped"] else 0.0
    advanced_info["total_reward"] = reward
    
    return reward, advanced_info

def compute_lift_reward(obs, achieved_goal, desired_goal, touch_values=None, prev_z=None, height_threshold=0.07):
    """
    简洁高效的抓取+抬升奖励函数：
    - 抓住奖励
    - 抬升奖励（z高于阈值）
    - 抓住后z越高奖励越大
    - 抓住但掉落惩罚
    """
    reward = 0.0
    z_pos = achieved_goal[2]
    grasped = False
    lifted = False
    drop_penalty = 0.0

    # 判断是否抓住（触点数量>10）
    if touch_values is not None:
        num_touched = np.sum(touch_values > 0.0)
        grasped = num_touched > 10

    # 抓住奖励
    if grasped:
        reward += 5.0
        # 抬升奖励
        if z_pos > height_threshold:
            lifted = True
            reward += 10.0 + 30.0 * (z_pos - height_threshold)  # 抬得越高奖励越大
    # 没抓住但z高，给一点探索奖励
    elif z_pos > height_threshold:
        reward += 1.0

    # 抓住后掉落惩罚
    if prev_z is not None and grasped and (prev_z - z_pos) > 0.05:
        drop_penalty = -10.0
        reward += drop_penalty

    info = {
        "grasped": grasped,
        "lifted": lifted,
        "z_pos": z_pos,
        "reward": reward,
        "drop_penalty": drop_penalty
    }
    return reward, info
