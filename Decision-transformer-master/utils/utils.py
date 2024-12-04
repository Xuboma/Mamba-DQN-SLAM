import argparse
import time


def read_ground_truth(gt_dict, gt_path):
    """
    获取位姿真值数据
    :param gt_path:真值数据路径
    :return:{时间戳：[位姿数据]}
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data_processed = list(map(lambda x: x.strip().split(' '), data))
    for item in data_processed:
        gt_dict[item[0]] = [pose for pose in item[1:]]


def get_next_frame(timestamp_path, current_frame):
    """
    获取下一帧的帧名
    :param timestamp_path: 时间戳路径
    :param current_frame: 当前帧名
    :return:
    """
    with open(timestamp_path, 'r') as f:
        ls = f.readlines()
        ls = list(map(lambda x: x.strip(), ls))
        current_frame_index = ls.index(current_frame)
        temp_index=current_frame_index + 1
        if temp_index == len(ls):
            return None
        next_frame = ls[temp_index]
        return next_frame



def write_my_params(img_name,sf,nl,nartio):
    with open('../my_params.txt', 'a') as f:
        f.write(f"{img_name},{sf},{nl},{nartio}\n")


def init(dataset_name, args):
    if dataset_name == 'MH01':
        args.gt_path = '../truth_deal/true_m1.txt'  # 设置真值路径
        args.image_path = '../data/MH01/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '../Examples/Monocular/EuRoC_TimeStamps/MH01.txt'  # 设置时间戳路径
        #1403636579763555584.png,1.2,8,0.8
        # write_my_params("1403636579763555584", 1.2, 8, 0.8)
    elif dataset_name == 'MH02':
        args.gt_path = '../truth_deal/true_m2.txt'
        args.image_path = '../data/MH02/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '../Examples/Monocular/EuRoC_TimeStamps/MH02.txt'  # 设置时间戳路径
    elif dataset_name == 'MH03':
        args.gt_path = '../truth_deal/true_m3.txt'
        args.image_path = '../data/MH03/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '../Examples/Monocular/EuRoC_TimeStamps/MH03.txt'  # 设置时间戳路径
    elif dataset_name == 'MH04':
        args.gt_path = '../truth_deal/true_m4.txt'
        args.image_path = '../data/MH04/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '../Examples/Monocular/EuRoC_TimeStamps/MH04.txt'  # 设置时间戳路径
    elif dataset_name == 'MH05':
        args.gt_path = '../truth_deal/true_m5.txt'
        args.image_path = '../data/MH05/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '../Examples/Monocular/EuRoC_TimeStamps/MH05.txt'  # 设置时间戳路径


if __name__ == '__main__':
    write_my_params("1403636579763555584",1.19,6,0.73)
