
from Utils.compute_rmse import compute_rmse_with_file



ref_file = f"/home/wwp/GNN-ORB_SLAM3-master/MH01_GT.tum"
est_file = f"/home/wwp/GNN-ORB_SLAM3-master/f_dataset-MH01_mono.txt"

# rmse = compute_rmse_with_file(est_file,ref_file)
rmse = compute_rmse_with_file(ref_file, est_file)
print(rmse)
