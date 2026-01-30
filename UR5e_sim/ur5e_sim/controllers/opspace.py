from typing import Optional, Tuple, Union

import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr


def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    x_err = x - x_des
    dx_err = dx

    # Apply gains.
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    return x_err + dx_err


def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    dw_max: float = 0.0,
) -> np.ndarray:
    # Compute error.
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = tr.quat_to_axisangle(quat_err)
    w_err = w

    # Apply gains.
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    # Limit maximum error.
    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)

    return ori_err + w_err


class OpSpaceController:
    def __init__(self, model, dof_ids):
        self.dof_ids = dof_ids
        self.nv = model.nv
        self.n_dof = len(dof_ids)

        # Pre-allocate buffers
        self.J_v = np.zeros((3, self.nv), dtype=np.float64)
        self.J_w = np.zeros((3, self.nv), dtype=np.float64)
        self.M = np.zeros((self.nv, self.nv), dtype=np.float64)
        self.damping = 1e-4 * np.eye(6)
        self.identity = np.eye(self.n_dof)

        # Pre-allocate reuse buffers to minimize allocation
        self.J = np.zeros((6, self.n_dof), dtype=np.float64)
        self.M_sub = np.zeros((self.n_dof, self.n_dof), dtype=np.float64)
        self.ddx_dw = np.zeros((6,), dtype=np.float64)
        self.tau = np.zeros((self.n_dof,), dtype=np.float64)
        self.eye_dof = np.eye(self.n_dof)
        self.eye_6 = np.eye(6)

    def __call__(
        self,
        model,
        data,
        site_id,
        pos: Optional[np.ndarray] = None,
        ori: Optional[np.ndarray] = None,
        joint: Optional[np.ndarray] = None,
        pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
        ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
        damping_ratio: float = 1.0,
        nullspace_stiffness: float = 0.5,
        max_pos_acceleration: Optional[float] = None,
        max_ori_acceleration: Optional[float] = None,
        gravity_comp: bool = True,
    ) -> np.ndarray:
        if pos is None:
            x_des = data.site_xpos[site_id]
        else:
            x_des = np.asarray(pos)
        if ori is None:
            xmat = data.site_xmat[site_id].reshape((3, 3))
            quat_des = tr.mat_to_quat(xmat)
        else:
            ori = np.asarray(ori)
            if ori.shape == (3, 3):
                quat_des = tr.mat_to_quat(ori)
            else:
                quat_des = ori
        if joint is None:
            q_des = data.qpos[self.dof_ids]
        else:
            q_des = np.asarray(joint)

        kp = np.asarray(pos_gains)
        kd = damping_ratio * 2 * np.sqrt(kp)
        kp_kv_pos = np.stack([kp, kd], axis=-1)

        kp = np.asarray(ori_gains)
        kd = damping_ratio * 2 * np.sqrt(kp)
        kp_kv_ori = np.stack([kp, kd], axis=-1)

        kp_joint = np.full((self.n_dof,), nullspace_stiffness)
        kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
        kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

        ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
        dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0

        # Get current state.
        q = data.qpos[self.dof_ids]
        dq = data.qvel[self.dof_ids]

        # Compute Jacobian of the eef site in world frame.
        mujoco.mj_jacSite(model, data, self.J_v, self.J_w, site_id)

        # Combine Jacobian parts without allocation
        # Reuse self.J buffer to avoid allocation
        self.J[:3, :] = self.J_v[:, self.dof_ids]
        self.J[3:, :] = self.J_w[:, self.dof_ids]
        J = self.J

        # Compute position PD control.
        x = data.site_xpos[site_id]
        dx = self.J_v[:, self.dof_ids] @ dq
        ddx = pd_control(x=x, x_des=x_des, dx=dx, kp_kv=kp_kv_pos, ddx_max=ddx_max)

        # Compute orientation PD control.
        quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
        if quat @ quat_des < 0.0:
            quat *= -1.0
        w = self.J_w[:, self.dof_ids] @ dq
        dw = pd_control_orientation(quat=quat, quat_des=quat_des, w=w, kp_kv=kp_kv_ori, dw_max=dw_max)

        # Compute inertia matrix in joint space.
        mujoco.mj_fullM(model, self.M, data.qM)
        # Use advanced indexing but write into pre-allocated buffer if possible
        # Numpy advanced indexing always returns a copy, so we can't avoid allocation easily here
        # unless we loop or use fancy indexing features.
        # But (6,6) copy is fast.
        M_sub = self.M[self.dof_ids, :][:, self.dof_ids]

        # Compute inertia matrix in task space.
        # Use solve instead of inv for numerical stability and performance
        M_inv = np.linalg.solve(M_sub, self.eye_dof)
        Mx_inv = J @ M_inv @ J.T

        # Solve for Mx explicitly is expensive and unstable; we use it for nullspace though.
        # For force calculation: (Mx_inv + damping) @ F_op = ddx_dw
        Mx_inv_damped = Mx_inv + self.damping

        # 1. Compute Operational Space Force directly using solve
        self.ddx_dw[:3] = ddx
        self.ddx_dw[3:] = dw

        F_op = np.linalg.solve(Mx_inv_damped, self.ddx_dw)
        tau = J.T @ F_op

        # 2. Compute Nullspace Control
        Mx = np.linalg.solve(Mx_inv_damped, self.eye_6)

        # Add joint task in nullspace.
        ddq = pd_control(x=q, x_des=q_des, dx=dq, kp_kv=kp_kv_joint, ddx_max=0.0)
        Jnull = M_inv @ J.T @ Mx
        tau += (self.identity - J.T @ Jnull.T) @ ddq

        if gravity_comp:
            tau += data.qfrc_bias[self.dof_ids]
        return tau

# Backward compatibility wrapper
def opspace(model, data, site_id, dof_ids, **kwargs):
    controller = OpSpaceController(model, dof_ids)
    return controller(model, data, site_id, **kwargs)
