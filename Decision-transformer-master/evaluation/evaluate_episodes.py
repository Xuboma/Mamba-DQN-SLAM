import os

import numpy as np
import torch
import time
from PIL import Image
from .rigid_transform_3D import rigid_transform_3D
from utils.utils import get_next_frame, write_my_params

# 初始化真值位姿数据与评估位姿数据为空np数组，用来存放已经计算的帧的真实位姿与评估位姿
gt_data_list = np.empty((0, 3))
estimate_data_list = np.empty((0, 3))
rmse=0.50001

# def get_next_frame(temp_next_frame_name_path):
#     global last_next_frame_name
#
#     while True:
#         current_frame_name = os.path.getmtime(temp_next_frame_name_path)
#         if current_frame_name != last_next_frame_name:
#             with open(temp_next_frame_name_path, 'r') as f:
#                 return f.readline().strip()
#         time.sleep(0.1)
def get_estimate_pose(result_path):
    """
    读取result.txt的位姿数据
    :param result_path:ORB-SLAM3 计算得到的位姿结果文件路径
    :return: 当前帧的位姿结果序列[帧ID，帧图片名，时间戳，平移数据，旋转数据]
    # ['6', '1403636580013555456', '1403636580.013556', '0.001630429', '0.043768696', '0.012056598', '-0.012066001', '0.002291820', '0.007707631', '0.999894857']
    """

    global last_modified_time

    data = []

    while True:
        current_modified_time = os.path.getmtime(result_path)  # 再次获取文件的最后修改时间
        if current_modified_time != last_modified_time:
            with open(result_path, 'r') as f:
                data = f.readline().strip().split(' ')
            if len(data) < 2:
                print("未读取到数据")
                continue
            print(f'已获得{data[1]}.png的位姿结果')
            last_modified_time = current_modified_time
            break
    return data


def set_params(num_params, num_episodes, params,current_frame_name):
    """
    将参数写入read.txt 与read_all.txt
    :param num_params:
    :param num_episodes:
    :param params:
    :return:
    """
    with open('../read.txt', 'w') as f, \
            open('../read_all.txt', 'a') as f_all:
        f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        # print('num_params, num_episodes', num_params, num_episodes)
        print(f"read.txt成功写入参数，参数为{params[0]} {params[1]} {params[2]}")
        print('wait for SLAM processing ...')
        print('=========================================')
    time.sleep(0.01)
    write_my_params(current_frame_name,params[0],params[1],params[2])


def get_state_and_reward(gt_dict,variant):
    """
    #estimate_pose  ['6', '1403636580013555456', '1403636580.013556', '0.001630429', '0.043768696', '0.012056598', '-0.012066001', '0.002291820', '0.007707631', '0.999894857']
    获取强化学习所需的state和reward
    这里state定义为图片，reward为1-rmse
    :param gt_dict: 真值位姿数据字典
    :return: state reward
    """
    estimate_pose = get_estimate_pose('../result.txt')
    # estimate_pose = get_estimate_pose('../temp_pose.txt')

    # 已获得位姿的帧
    prior_frame_name = estimate_pose[1]

    # 下一个需要计算参数的帧
    # current_frame_name = get_next_frame('../temp_next_frame_name.txt')
    current_frame_name = get_next_frame(variant['timestamp_path'], prior_frame_name)
    if current_frame_name==None:
        return current_frame_name,None, None
    print(f"正在计算{current_frame_name}的参数")

    temp_cur_frame_name = current_frame_name[:-4] + '0000'

    if temp_cur_frame_name not in gt_dict.keys():
        # 当真实数据中不存在该帧的gt值时，沿用上一次的rmse
        im = Image.open(variant['image_path'] + current_frame_name + '.png')
        im = torch.from_numpy(np.array(im.resize((64, 64))))
        return current_frame_name,im, 1-rmse

    errs = get_error(estimate_pose, gt_dict)
    reward = 1 - errs

    im = Image.open(variant['image_path'] + current_frame_name + '.png')
    im = torch.from_numpy(np.array(im.resize((64, 64))))

    print(f"[{list(gt_dict.keys()).index(temp_cur_frame_name) + 1}/{len(gt_dict)}]正在计算{current_frame_name}.png的参数...")
    return current_frame_name,im, reward


def compute_rmse(gt_data, estimate_data):
    """
    计算绝对轨迹误差rmse（这里只计算平移位姿）
    :param gt_data: numpy数组[[x,y,z],...]
    :param estimate_data: numpy数组[[x,y,z],...]
    :return:
    """
    ground_temp = np.transpose(gt_data)
    estimate_temp = np.transpose(estimate_data)
    # Recover R and t
    ret_R, ret_t = rigid_transform_3D(estimate_temp, ground_temp)
    # Compare the recovered R and t with the original
    estimate_temp_val = (ret_R @ estimate_temp) + ret_t
    err = ground_temp - estimate_temp_val
    err = err * err  # 点乘
    err = np.sum(err)
    rmse = np.sqrt(err / len(estimate_data))  # 均方根误差
    return rmse


def get_error(estimate_pose, gt_dict):
    """
    获得系统不确定度error(即ATE rmse)
    :param estimate_pose:对应帧的评估位姿数据[[x,y,z],...]
    :param gt_dict:真值位姿数据字典
    :return:
    """

    global gt_data_list, estimate_data_list, rmse

    timestamp = estimate_pose[2].replace(".", "")[:-1] + "0000"

    # 如果当前帧不在gt_data中则返回上一次的rmse=0.50001
    if timestamp not in gt_dict.keys():
        return rmse

    pose = list(map(float, gt_dict[timestamp]))  # 字符串转float
    estimate_pose = list(map(float, estimate_pose))


    # 将真实平移位姿加入到真值位姿数组
    gt_data_list = np.concatenate((gt_data_list, [pose[0:3]]), axis=0)

    # 将评估平移位姿加入到评估位姿数组
    estimate_data_list = np.concatenate((estimate_data_list, [estimate_pose[3:6]]), axis=0)


    # 当保存的位姿数据个数大于2时再开始计算rmse
    if len(estimate_data_list) > 2:
        rmse = compute_rmse(gt_data_list, estimate_data_list)
    else:
        rmse = 0.50001

    return rmse


def evaluate_episode_rtg(env, act_dim, model, scale=1000., device='cpu', target_return=None, gt_dict=None,variant=None):

    model.eval()

    num_params = 0
    num_episodes = 1

    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)  # 初始化动作
    rewards = torch.zeros(0, device=device, dtype=torch.float32)  #初始化奖励

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)



    global last_modified_time
    last_modified_time = 0

    global last_next_frame_name #保存时间
    last_next_frame_name = 0

    episode_return, episode_length = 0, 0
    while True:
        current_frame_name, states, reward = get_state_and_reward(gt_dict,variant)
        if current_frame_name==None and states==None and reward==None:
            break

        feature = states.clone().detach().to(device=device, dtype=torch.float32).requires_grad_(True)

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        reward = torch.tensor(reward, device=device, dtype=torch.float32)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            feature.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()
        rewards[-1] = reward
        action_max = np.argmax(abs(action))
        a = (action_max).astype(np.int32)
        params = env.parameter_space[a]
        num_params = num_params + 1
        num_episodes = num_episodes + 1

        # 将获取的参数写入read.txt 与read_all.txt
        set_params(num_params, num_episodes, params,current_frame_name)

        pred_return = target_return[0, -1] - (reward / scale)

        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (num_params + 1)],
                              dim=1)

        episode_return += reward
        episode_length += 1

    return episode_return, episode_length
