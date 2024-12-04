import math
import os

import numpy as np
import time
from PIL import Image

from Utils.rigid_transform3D import rigid_transform_3D
from Utils.utils import get_next_frame
from torchvision import  transforms
from Utils.compute_rmse import compute_rmse_with_file

# 初始化真值位姿数据与评估位姿数据为空np数组，用来存放已经计算的帧的真实位姿与评估位姿
gt_data_list = np.empty((0, 3))
estimate_data_list = np.empty((0, 3))
rmse=0.50001


def get_estimate_pose(result_path):
    """
    读取result.txt的位姿数据
    :param result_path:ORB-SLAM3 计算得到的位姿结果文件路径
    :return: 当前帧的位姿结果序列[帧ID，帧图片名，时间戳，平移数据，旋转数据]
    # ['6', '1403636580013555456', '1403636580.013556', '0.001630429', '0.043768696', '0.012056598', '-0.012066001', '0.002291820', '0.007707631', '0.999894857']
    """

    global last_modified_time
    last_modified_time = {}
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


def write_my_params(frame_name, sf, nl, nn):
    file_path = './my_params.txt'

    # 检查文件是否存在且不为空
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                last_frame_name = last_line.split(',')[0]
                # 如果frame_name和最后一行的第一列的内容一样，则不写入
                if frame_name == last_frame_name:
                    return
    # 写入文件
    with open(file_path, 'a') as f:
        f.write(f'{frame_name},{sf},{nl},{nn}\n')


# def write_my_params(frame_name,sf,nl,nn):
#     with open('./my_params.txt', 'a') as f:
#         f.write(f'{frame_name},{sf},{nl},{nn}\n')


def set_params(num_params, num_episodes, params, args):
    """
    将参数写入read.txt 与read_all.txt
    :param num_params:
    :param num_episodes:
    :param params:
    :return:
    """
    with open(args.read_path, 'w') as f, \
            open(args.read_all_path, 'a') as f_all:
        f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        # print('num_params, num_episodes', num_params, num_episodes)
        print(f"read.txt成功写入参数，参数为{params[0]} {params[1]} {params[2]}")
        print('wait for SLAM processing ...')
        print('=========================================')
        succ = True
    time.sleep(0.01)

    return succ



def process_image_with_mobilenet(image, model):
    """
    使用预训练的 MobileNet 模型处理图像
    :param image: 需要处理的图像
    :param model: 预训练的 MobileNet 模型
    :return: 处理后的图像
    """
    # 调整图像的大小，然后转换为 Tensor
    image = image.resize((224, 224))  # 注意，MobileNet V2 需要的输入大小是 224x224
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    # 预处理图像
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    input_tensor = preprocess(image)

    # 使用 MobileNet 模型处理图像
    query_features = model(input_tensor)

    # 将特征重塑为64*64的大小
    query_features = query_features.view(-1, 64, 64)

    # 移除维度大小为1的维度
    query_features = query_features.squeeze(0)

    # 将结果转换为numpy数组
    processed_image = query_features.detach().numpy()

    return processed_image

def append_to_result_file(data):
    with open('/home/wwp/mxb/ORB_SLAM3-master/f_mono_pose_all.txt', 'a') as f:
        cleaned_items = [item.rstrip() for item in data]
        line = ' '.join(cleaned_items)  # 使用空格连接清理后的每个项目
        f.write(line + '\n')  # 将处理后的行写入文件，但不在行末添加额外的换行符

def count_lines_in_file(file_path):
    with open(file_path, 'r') as f:
        num_lines = len(f.readlines())
    return num_lines

def get_state_and_reward(gt_dict, args):
    """
    #estimate_pose  ['6', '1403636580013555456', '1403636580.013556', '0.001630429', '0.043768696', '0.012056598', '-0.012066001', '0.002291820', '0.007707631', '0.999894857']
    获取强化学习所需的state和reward
    这里state定义为图片，reward为1-rmse
    :param gt_dict: 真值位姿数据字典
    :return: state reward
    """
    variant = vars(args)
    estimate_pose = get_estimate_pose(args.result_path)

    # 将相机Pose放在文件里用于计算新的奖励
    # append_to_result_file(estimate_pose[2:])
    # ref_file = args.ref_path
    # est_file= args.f_result_all_path
    # num_lines = count_lines_in_file(est_file)
    # # 在这里计算RMSE从开始到现在的RMSE
    # Reward = 8
    # if num_lines > 50:
    #     rmse_evo = compute_rmse_with_file(ref_file, est_file)
    #     Reward = -math.log(rmse_evo)
    #     print("reward:", Reward)

    prior_frame_name = estimate_pose[1]

    # 下一个需要计算参数的帧
    # current_frame_name = get_next_frame('../temp_next_frame_name.txt')
    current_frame_name = get_next_frame(variant['timestamp_path'], prior_frame_name)
    if current_frame_name==None:
        return current_frame_name, None, None
    print(f"正在计算下一帧{current_frame_name}的参数")
    temp_cur_frame_name = current_frame_name[:-4] + '0000'

    if temp_cur_frame_name not in gt_dict.keys():
        # 当真实数据中不存在该帧的gt值时，沿用上一次的rmse
        im = Image.open(variant['image_path'] + current_frame_name + '.png')
        # 这是用resnet处理后的
        # im = process_image_with_mobilenet(im, model)
        # 这是原来的
        # im = torch.from_numpy(np.array(im.resize((224, 224))))
        succ = True
        return current_frame_name, im, 1-rmse, succ
        # return current_frame_name, im, Reward, succ
    errs = get_error(estimate_pose, gt_dict)
    # print("误差：", errs)
    reward = 1 - errs
    im = Image.open(variant['image_path'] + current_frame_name + '.png')
    # 这是用resnet处理后的
    # im = process_image_with_mobilenet(im, model)
    # 这是原来的
    # im = torch.from_numpy(np.array(im.resize((224, 224))))

    print(f"[{list(gt_dict.keys()).index(temp_cur_frame_name) + 1}/{len(gt_dict)}]正在计算{current_frame_name}.png的参数...")
    succ = True

    return current_frame_name, im, reward, succ
    # return current_frame_name, im, Reward, succ





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
