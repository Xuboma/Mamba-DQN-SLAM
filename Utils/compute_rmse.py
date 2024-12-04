from evo.core.metrics import PoseRelation
from evo.tools import file_interface
import numpy as np
from evo.core.result import Result
from evo.core.trajectory import PosePath3D, Plane, PoseTrajectory3D
from evo.core import lie_algebra, sync, metrics
import typing


def ape(traj_ref: PosePath3D, traj_est: PosePath3D,
        pose_relation: metrics.PoseRelation, align: bool = False,
        correct_scale: bool = False, n_to_align: int = -1,
        align_origin: bool = False, ref_name: str = "reference",
        est_name: str = "estimate",
        change_unit: typing.Optional[metrics.Unit] = None,
        project_to_plane: typing.Optional[Plane] = None) -> Result:
    """
    代码来自官方evo
    """

    # Align the trajectories.
    only_scale = correct_scale and not align
    alignment_transformation = None
    if align or correct_scale:

        alignment_transformation = lie_algebra.sim3(
            *traj_est.align(traj_ref, correct_scale, only_scale, n=n_to_align))
    elif align_origin:

        alignment_transformation = traj_est.align_origin(traj_ref)

    # Projection is done after potential 3D alignment & transformation steps.
    if project_to_plane:
        traj_ref.project(project_to_plane)
        traj_est.project(project_to_plane)

    # Calculate APE.
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    if change_unit:
        ape_metric.change_unit(change_unit)

    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"
    if (align or correct_scale) and n_to_align != -1:
        title += " (aligned poses: {})".format(n_to_align)

    if project_to_plane:
        title += f"\n(projected to {project_to_plane.value} plane)"

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, PoseTrajectory3D):
        seconds_from_start = np.array(
            [t - traj_est.timestamps[0] for t in traj_est.timestamps])
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)
        ape_result.add_np_array("distances_from_start", traj_ref.distances)
        ape_result.add_np_array("distances", traj_est.distances)

    if alignment_transformation is not None:
        ape_result.add_np_array("alignment_transformation_sim3",
                                alignment_transformation)

    return ape_result


def compute_rmse_with_file(est_file, ref_file):
    # 读取位姿文件


    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    # 设置为旋转误差
    pose_relation = PoseRelation.translation_part
    # 轨迹对齐，轨迹关联
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = ape(traj_ref=traj_ref, traj_est=traj_est,
                 pose_relation=pose_relation, align=True,
                 correct_scale=True)

    return result.stats['rmse']




if __name__ == '__main__':
    #est_file = "/home/mxb/DTQN/mydtqn/ORB_SLAM3/结果/MH01/测试/MH01/f_dataset-MH01_mono.txt"
    ref_file = "/home/wwp/mxb/ORB_SLAM3-master/结果/MH01/0.04/gt_MH01.tum"
    est_file= "/home/wwp/mxb/ORB_SLAM3-master/结果/MH01/0.04/f_dataset-MH01_mono.txt"
    rmse = compute_rmse_with_file(ref_file, est_file)
    print(rmse)
