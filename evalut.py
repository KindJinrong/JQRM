from scipy.spatial.transform import Rotation as R
import numpy as np
import math
device = "cpu"
def rot_error(r_gt, r_est):
    r_gt_np = r_gt.to(device).numpy()
    r_est_np = r_est.to(device).numpy()
    
    # 使用 scipy 计算旋转误差
    relative_rotation = np.dot(np.linalg.inv(r_gt_np), r_est_np)
    angle = R.from_matrix(relative_rotation).magnitude()
    
    return angle * 180 / math.pi

def position_error(r_gt,r_est):
    # r_gt:True Euclidean Distance
    dis = np.sqrt(np.mean((r_gt.to(device).numpy() - r_est.to(device).numpy())**2))
    return dis