import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Dict

class SplitObsTransformerEncoder(nn.Module):
    """
    Encodes observations using a Set Transformer for the obstacles.
    Splits the state into:
        1. Robot State (Indices 0-19): MLP
        2. Env Scalars (Indices 19-25, Block+Target): MLP
        3. Obstacles (Indices 25+, 128*7): Transformer (Permutation Invariant)
    """

    # Robot Branch (same as SplitObsEncoder)
    robot_hidden_dims = [32, 64, 128, 256, 128]
    # Env Scalar Branch
    scalar_hidden_dims = [32, 64]
    # Obstacle Branch
    obstacle_embed_dim = 64
    transformer_num_heads = 4
    transformer_num_layers = 2
    transformer_mlp_dim = 128

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> jnp.ndarray:

        if "state" not in observations:
            raise KeyError("SplitObsTransformerEncoder requires 'state' in observations.")

        state = observations["state"]

        # Handle time/batch dims
        # Flatten time into batch if present
        is_sequence = False
        original_shape = state.shape
        if state.ndim == 3: # (B, T, C)
             B, T, C = state.shape
             state = rearrange(state, "B T C -> (B T) C")
             is_sequence = True
        elif state.ndim == 2: # (T, C) -> (T, C) or (B, C) treated as batch
             pass

        # Split State
        # Robot: 0-19
        # Scalars: 19-25 (Block: 3, Target: 3)
        # Obstacles: 25+ (128 * 7)
        robot_state = state[..., :19]
        scalar_state = state[..., 19:25]
        obstacle_state = state[..., 25:]

        # --- 1. Robot Branch ---
        x_robot = robot_state
        for size in self.robot_hidden_dims:
            x_robot = nn.Dense(size)(x_robot)
            x_robot = nn.LayerNorm()(x_robot)
            x_robot = nn.relu(x_robot)

        # --- 2. Scalar Branch ---
        x_scalar = scalar_state
        for size in self.scalar_hidden_dims:
            x_scalar = nn.Dense(size)(x_scalar)
            x_scalar = nn.LayerNorm()(x_scalar)
            x_scalar = nn.relu(x_scalar)

        # --- 3. Obstacle Branch (Transformer) ---
        # Reshape to (Batch, Num_Obstacles, Feature_Dim)
        # 128 obstacles, 7 features each
        batch_size = obstacle_state.shape[0]
        x_obs = obstacle_state.reshape(batch_size, 128, 7)

        # Linear Projection
        x_obs = nn.Dense(self.obstacle_embed_dim)(x_obs) # (B, 128, 64)

        # Transformer Encoder Blocks
        for _ in range(self.transformer_num_layers):
            # Self-Attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.transformer_num_heads,
                qkv_features=self.obstacle_embed_dim,
                out_features=self.obstacle_embed_dim,
            )(x_obs, x_obs)
            x_obs = nn.LayerNorm()(x_obs + attn_out)

            # MLP Block
            mlp_out = nn.Dense(self.transformer_mlp_dim)(x_obs)
            mlp_out = nn.relu(mlp_out)
            mlp_out = nn.Dense(self.obstacle_embed_dim)(mlp_out)
            x_obs = nn.LayerNorm()(x_obs + mlp_out)

        # Global Pooling (Permutation Invariance)
        # Max pooling is often good for detecting collision "presence"
        x_obs = jnp.max(x_obs, axis=1) # (B, 64)

        # --- 4. Combine ---
        combined = jnp.concatenate([x_robot, x_scalar, x_obs], axis=-1)

        # Final projection to match typical embedding size if needed,
        # but pure concatenation is fine.
        # Current size: 128 + 64 + 64 = 256. (Similar to original output)

        if stop_gradient:
            combined = jax.lax.stop_gradient(combined)

        if is_sequence:
             combined = combined.reshape(B, T, -1)

        return combined
