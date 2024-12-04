import argparse
import os.path

from Utils.compute_rmse import compute_rmse_with_file

parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MH01', help="选择数据集['MH01','MH02'...]")
args=parser.parse_args()

ref_file = f"/home/wwp/mxb/ORB_SLAM3-master/true_tum/gt_{args.dataset}.tum"
est_file = f"/home/wwp/mxb/ORB_SLAM3-master/f_dataset-{args.dataset}_mono.txt"
est_file2 = f"/home/wwp/mxb/ORB_SLAM3-master/f_dataset-{args.dataset}_512_mono.txt"

# rmse = compute_rmse_with_file(est_file,ref_file)
rmse = compute_rmse_with_file(ref_file, est_file2)
rmse2 = compute_rmse_with_file(est_file2, ref_file)
if not os.path.exists('./results'):
    os.mkdir('./results')
result_dir=f'./results/{args.dataset}.txt'
with open(result_dir,'a+',encoding='utf-8')as f:
    f.write(f'{args.number} 反:{rmse} 正：{rmse2}\n')
print(rmse)
