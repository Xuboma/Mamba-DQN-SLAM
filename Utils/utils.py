
# 设置数据集的路径
def init(dataset_name, args):

    args.read_path = '/home/wwp/mxb/ORB_SLAM3-master/read.txt'  # 设置参数路径路径
    args.read_all_path = '/home/wwp/mxb/ORB_SLAM3-master/read_all.txt'  # 设置参数路径路径
    args.result_path = '/home/wwp/mxb/ORB_SLAM3-master/result.txt'
    args.f_result_all_path = '/home/wwp/mxb/ORB_SLAM3-master/f_mono_pose_all.txt'

    if dataset_name == 'MH01':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_m1.txt'  # 设置真值路径
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/MH01/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/MH01.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_MH01.tum'
    elif dataset_name == 'MH02':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_m2.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/MH02/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/MH02.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_MH02.tum'
    elif dataset_name == 'MH03':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_m3.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/MH03/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/MH03.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_MH03.tum'
    elif dataset_name == 'MH04':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_m4.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/MH04/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/MH04.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_MH04.tum'
    elif dataset_name == 'MH05':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_m5.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/MH05/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/MH05.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_MH05.tum'
    elif dataset_name == 'V101':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v101.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V101/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V101.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V101.tum'
    elif dataset_name == 'V102':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v102.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V102/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V102.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V102.tum'
    elif dataset_name == 'V103':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v103.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V103/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V103.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V103.tum'
    elif dataset_name == 'V201':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v201.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V201/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V201.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V201.tum'
    elif dataset_name == 'V202':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v202.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V202/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V202.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V202.tum'
    elif dataset_name == 'V203':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/true_v203.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/V203/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/EuRoC_TimeStamps/V203.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_V203.tum'

    elif dataset_name == 'corridor1':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_c1.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/corridor1/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_corridor1.tum'
    elif dataset_name == 'corridor2':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_c2.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/corridor2/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-corridor2_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_corridor2.tum'
    elif dataset_name == 'corridor3':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_c3.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/corridor3/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-corridor3_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_corridor3.tum'
    elif dataset_name == 'corridor4':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_c4.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/corridor4/mav0/cam0/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-corridor4_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_corridor4.tum'
    elif dataset_name == 'corridor5':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_c5.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/corridor5/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-corridor5_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_corridor5.tum'
    elif dataset_name == 'room1':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r1.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room1/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room1_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room1.tum'
    elif dataset_name == 'room2':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r2.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room2/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room2_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room2.tum'
    elif dataset_name == 'room3':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r3.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room3/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room3_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room3.tum'
    elif dataset_name == 'room4':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r4.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room4/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room4_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room4.tum'
    elif dataset_name == 'room5':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r5.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room5/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room5_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room5.tum'
    elif dataset_name == 'room6':
        args.gt_path = '/home/wwp/mxb/ORB_SLAM3-master/truth_deal/truth_r6.txt'
        args.image_path = '/home/wwp/mxb/ORB_SLAM3-master/data/room6/mav0/cam1/data/'  # 设置图片路径
        args.timestamp_path = '/home/wwp/mxb/ORB_SLAM3-master/Examples/Monocular/TUM_TimeStamps/dataset-room6_512.txt'  # 设置时间戳路径
        args.ref_path = '/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_room6.tum'




def read_ground_truth(gt_dict, gt_path):
    """
    获取位姿真值数据
    :param gt_path:真值数据路径
    :return:{时间戳：[位姿数据]}
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        #print(data)
    data_processed = list(map(lambda x: x.strip().split(' '), data))
    for item in data_processed:
        gt_dict[item[0]] = [pose for pose in item[1:]]
    print("---------0.获取真实位姿成功---------")


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
        temp_index = current_frame_index + 1
        if temp_index == len(ls):
            return None
        next_frame = ls[temp_index]
        return next_frame


def get_true_traj(path):
    true_traj = []
    with open(path, 'r') as f_true:
        while True:
            line_true = f_true.readline()
            true_traj.append(line_true)
            if not line_true:
                break
    print('真实轨迹位姿获取成功：true_traj')
    # print(true_traj)
    return true_traj



