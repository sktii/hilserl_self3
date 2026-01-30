import os
import jax
import jax.numpy as jnp
import numpy as np
import glfw
import gymnasium as gym
import cv2

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper # [FIX] 移除 ChunkingWrapper 引用 (可選)
from serl_launcher.networks.reward_classifier import load_classifier_func

from examples_UR5e.experiments.config import DefaultTrainingConfig

from ur5e_sim.envs.ur5e_stack_gym_env import UR5eStackCubeGymEnv

class EnvConfig(DefaultEnvConfig):
    # Curriculum parameters (can be overridden by TrainConfig.get_environment kwargs or CLI)
    TASK_STAGE = "full"
    LOAD_STATE_DIR = None
    SAVE_STATE_DIR = None

    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "left": {
            "serial_number": "127122270146",
            "dim": (128, 128),
            "exposure": 40000,
        },
        "wrist": {
            "serial_number": "127122270350",
            "dim": (128, 128),
            "exposure": 40000,
        },
        "right": {
            "serial_number": "none",
            "dim": (128, 128),
            "exposure": 40000,
        },
    }
    def crop_and_resize(img):
        return cv2.resize(img, (128, 128)) 

    IMAGE_CROP = {
        "left": crop_and_resize,
        "wrist": crop_and_resize,
        "right": crop_and_resize,
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.045, 0.18, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 260
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

        # Initialize gripper state to 'open'
        self.gripper_state = 'open'
        self.intervened = False
        self.action_length = 0.3
        self.current_action = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.flag = False
        self.last_gripper_pos = 0.0
        # self.last_semicolon_state = 0 # Handled in callback
        # self.last_l_state = 0 # Handled in callback

        # Key states for callback
        self.pressed_keys = {
            'w': False, 'a': False, 's': False, 'd': False,
            'j': False, 'k': False
        }
        self.callback_set = False

    def _key_callback(self, window, key, scancode, action, mods):
        # Update pressed keys
        if key == glfw.KEY_W:
            self.pressed_keys['w'] = (action != glfw.RELEASE)
        elif key == glfw.KEY_A:
            self.pressed_keys['a'] = (action != glfw.RELEASE)
        elif key == glfw.KEY_S:
            self.pressed_keys['s'] = (action != glfw.RELEASE)
        elif key == glfw.KEY_D:
            self.pressed_keys['d'] = (action != glfw.RELEASE)
        elif key == glfw.KEY_J:
            self.pressed_keys['j'] = (action != glfw.RELEASE)
        elif key == glfw.KEY_K:
            self.pressed_keys['k'] = (action != glfw.RELEASE)

        # Toggle Logic (Rising Edge)
        elif key == glfw.KEY_L and action == glfw.PRESS:
            self.flag = True

        elif key == glfw.KEY_SEMICOLON and action == glfw.PRESS:
            self.intervened = not self.intervened
            self.env.intervened = self.intervened
            # Sync Logic
            if self.intervened and self.gripper_enabled:
                 if self.last_gripper_pos > 0.05:
                      self.gripper_state = 'close'
                 else:
                      self.gripper_state = 'open'
                 print(f"Intervention ON. Synced gripper to: {self.gripper_state} (pos={self.last_gripper_pos:.3f})")
            else:
                 print(f"Intervention toggled: {self.intervened}")

    def _setup_callback(self):
        """Lazily setup GLFW callback"""
        if self.env.render_mode != "human":
            return

        if not hasattr(self, '_cached_window'):
            self._cached_window = None
            if hasattr(self.env, "_viewer") and self.env._viewer:
                if hasattr(self.env._viewer, "viewer") and self.env._viewer.viewer:
                    if hasattr(self.env._viewer.viewer, "window"):
                        self._cached_window = self.env._viewer.viewer.window
                elif hasattr(self.env._viewer, "window"):
                    self._cached_window = self.env._viewer.window

        window = self._cached_window
        if window is not None and not self.callback_set:
            glfw.set_key_callback(window, self._key_callback)
            self.callback_set = True
            print("Intervention: Custom key callback installed.")

    def action(self, action: np.ndarray) -> np.ndarray:
        self._setup_callback()

        # Update current action vector from state
        self.current_action[:3] = [
            int(self.pressed_keys['w']) - int(self.pressed_keys['s']), # x
            int(self.pressed_keys['a']) - int(self.pressed_keys['d']), # y
            int(self.pressed_keys['j']) - int(self.pressed_keys['k']), # z
        ]
        self.current_action[:3] *= self.action_length
        expert_a = self.current_action.copy()

        if self.gripper_enabled:
            # Handle gripper toggle logic
            if self.flag:
                if self.gripper_state == 'open':
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
                self.flag = False
            
            # Generate gripper action based on state
            if self.gripper_state == 'close':
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
            else:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))

            if self.env.action_space.shape[0] == 4:
                 expert_a = np.concatenate((expert_a[:3], gripper_action), axis=0)
            elif self.env.action_space.shape[0] == 7:
                 expert_a = np.concatenate((expert_a[:3], np.array([0,0,0,1]), gripper_action), axis=0)

        # Action Masking
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # Determine which action to return
        if self.intervened:
            return expert_a, True
        else:
            # Sync state: observe ENV state (physical) to update self.gripper_state
            # Using phys_gripper_pos from info is safer than command in obs
            if self.gripper_enabled:
                # Threshold for physical joint (0-1 normalized).
                # 0 is open (~0.03), 1 is closed.
                # Threshold > 0.05 implies intent to close or successful close.
                if self.last_gripper_pos > 0.05:
                    self.gripper_state = 'close'
                else:
                    self.gripper_state = 'open'
            
            return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # Capture actual physical gripper position from info for next sync
        try:
            val = None
            if "phys_gripper_pos" in info:
                val = info["phys_gripper_pos"]
            elif "state" in obs:
                 # Fallback to obs (command) if physical info missing
                if "ur5e/gripper_pos" in obs["state"]:
                    val = obs["state"]["ur5e/gripper_pos"]

            if val is not None:
                 if hasattr(val, "__getitem__") and hasattr(val, "__len__") and len(val) > 0:
                     self.last_gripper_pos = val[0]
                 else:
                     self.last_gripper_pos = val
        except Exception:
            pass

        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.intervened = False
        self.env.intervened = False
        self.gripper_state = 'open'
        self.last_gripper_pos = 0.0

        # Initial capture
        try:
            val = None
            if "phys_gripper_pos" in info:
                val = info["phys_gripper_pos"]
            elif "state" in obs:
                if "ur5e/gripper_pos" in obs["state"]:
                    val = obs["state"]["ur5e/gripper_pos"]

            if val is not None:
                 if hasattr(val, "__getitem__") and len(val) > 0:
                     self.last_gripper_pos = val[0]
                 else:
                     self.last_gripper_pos = val
        except Exception:
            pass

        return obs, info


class TrainConfig(DefaultTrainingConfig):
    image_keys = []
    classifier_keys = ["left", "wrist", "right"]
    proprio_keys = ["ur5e/tcp_pos", "ur5e/tcp_vel", "ur5e/joint_pos", "ur5e/joint_vel", "ur5e/gripper_pos", "block_pos_rel", "target_pos_rel", "cube_grasped", "obstacle_state"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 200
    encoder_type = "mlp"
    # setup_mode = "single-arm-fixed-gripper"
    setup_mode = "single-arm-learned-gripper"
    replay_buffer_capacity = 20000000

    def get_environment(self, fake_env=False, save_video=False, classifier=False, render_mode="human",
                        task_stage=None, load_state_dir=None, save_state_dir=None, model_part=None, success_buffer_size=100):
        # Allow overriding config defaults via arguments
        env_config = EnvConfig()

        # Map model_part to task_stage if provided (compatibility)
        if model_part and not task_stage:
            task_stage = model_part

        if task_stage: env_config.TASK_STAGE = task_stage
        if load_state_dir: env_config.LOAD_STATE_DIR = load_state_dir
        if save_state_dir: env_config.SAVE_STATE_DIR = save_state_dir

        # User requested 18 FPS to reduce system load (simulated FPS cap)
        env = UR5eStackCubeGymEnv(
            render_mode=render_mode,
            image_obs=False,
            hz=18,
            config=env_config,
            task_stage=env_config.TASK_STAGE,
            load_state_dir=env_config.LOAD_STATE_DIR,
            save_state_dir=env_config.SAVE_STATE_DIR,
            success_buffer_size=success_buffer_size
        )

        # NOTE: Classifier is force disabled here based on previous code snippets?
        # But 'classifier' arg comes in.
        # Ideally we respect the arg, but user wants to fix lag.
        # I will use the arg but add warmup.
        # classifier=False # This line was overriding the arg in previous snippet!

        if not fake_env:
            env = KeyBoardIntervention2(env)
            pass

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # === [FIX] 註解掉 ChunkingWrapper 以解決漸進式 Lag ===
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # =================================================

        # classifier=False # Removed override
        if classifier:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            # Warmup to prevent lag during interaction
            print("Compiling reward classifier...")
            dummy_obs = env.observation_space.sample()
            # Ensure dummy obs has correct keys for classifier
            classifier_func(dummy_obs)
            print("Classifier compiled.")

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                pred = sigmoid(classifier_func(obs))
                if hasattr(pred, 'shape') and len(pred.shape) > 0:
                    pred = pred.flatten()[0]
                return int(float(pred) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env