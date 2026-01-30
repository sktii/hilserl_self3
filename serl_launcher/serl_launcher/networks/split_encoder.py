from typing import Dict, Iterable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from serl_launcher.networks.mlp import MLP, MLPResNetBlock

class SplitObsEncoder(nn.Module):
    """
    Encodes observations by splitting the state into Robot and Env parts,
    processing them with separate MLPs, and concatenating the results.

    Expected Input:
        observations (Dict): Must contain key "state".
        "state" should be a flattened vector where:
            - First 19 elements are Robot State (tcp(3)+vel(3)+grip(1)+jpos(6)+jvel(6)).
            - The rest are Env State (block_pos, target_pos, obstacle_state...).
    """

    # Architecture params hardcoded as per user request
    # Robot Branch: [32, 64]
    robot_hidden_dims = [32, 64]
    # Env Branch: [64, 128, 64] using ResNet blocks
    env_hidden_dims = [64, 128, 64]

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False, # Kept for API compatibility with EncodingWrapper
    ) -> jnp.ndarray:

        # 1. Extract State
        # API Compatibility: EncodingWrapper expects 'state' in observations
        if "state" not in observations:
            raise KeyError("SplitObsEncoder requires 'state' in observations.")

        state = observations["state"]

        # 2. Handle Time/Batch dimensions
        # EncodingWrapper logic:
        # if len(state.shape) == 2: rearrange(state, "T C -> (T C)")
        # if len(state.shape) == 3: rearrange(state, "B T C -> B (T C)")

        if state.ndim == 2: # (T, C) -> (T*C)
             state = rearrange(state, "T C -> (T C)")
             # Now state is 1D (C)
        elif state.ndim == 3: # (B, T, C) -> (B, T*C)
             state = rearrange(state, "B T C -> B (T C)")
             # Now state is 2D (B, C)

        # If obs_horizon=1, T=1, so C is preserved.

        # 3. Split
        # New Layout (Total 53):
        # Robot (22): tcp_pos(3) + tcp_vel(6) + joint_pos(6) + joint_vel(6) + gripper_pos(1) -> Indices 0-22
        # Env Scalars (7): block_pos_rel(3) + target_pos_rel(3) + cube_grasped(1) -> Indices 22-29
        # Obstacles (24): 2 * 12 features -> Indices 29-53

        robot_state = state[..., :22]
        env_scalars = state[..., 22:29]
        obstacle_state = state[..., 29:]

        # 4. Robot Branch
        # Dense -> LN -> ReLU
        x_robot = robot_state
        for size in self.robot_hidden_dims:
            x_robot = nn.Dense(size)(x_robot)
            x_robot = nn.LayerNorm()(x_robot)
            x_robot = nn.relu(x_robot)

        # 5. Env Scalars Branch
        # Process scalars using ResNet blocks
        x_env = env_scalars
        for size in self.env_hidden_dims:
            x_env = MLPResNetBlock(
                features=size,
                act=nn.relu,
                use_layer_norm=True
            )(x_env, train=train)

        # 6. Obstacle Branch (PointNet: Permutation Invariant)
        # Reshape to (Batch, N_obstacles, Features) -> (..., N, 12)
        feat_dim = 12
        n_obs = int(obstacle_state.shape[-1] // feat_dim)
        obs_reshaped = obstacle_state.reshape(obstacle_state.shape[:-1] + (n_obs, feat_dim))

        # Shared MLP (applied to each obstacle independently)
        # Obstacle Branch: PointNet style [64, 64] -> MaxPool
        obstacle_hidden_dims = [64, 64]

        x_obs = obs_reshaped
        for size in obstacle_hidden_dims:
            x_obs = nn.Dense(size)(x_obs)
            x_obs = nn.LayerNorm()(x_obs)
            x_obs = nn.relu(x_obs)

        # Max Pooling over Obstacles (Permutation Invariance)
        # Reduce along the second-to-last dimension (N_obstacles)
        x_obs = jnp.max(x_obs, axis=-2)

        # 7. Concatenate
        combined = jnp.concatenate([x_robot, x_env, x_obs], axis=-1)

        if stop_gradient:
            combined = jax.lax.stop_gradient(combined)

        return combined
