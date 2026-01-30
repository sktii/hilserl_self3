from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import os
import glfw

# Prevent JAX from hogging GPU memory, allowing MuJoCo EGL to run smoothly
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"

import gym
import gymnasium # Need gymnasium.spaces for SERL compatibility
import mujoco
import numpy as np
from gym import spaces as gym_spaces # Keep gym spaces for legacy compat
from gymnasium import spaces as gymnasium_spaces # Use gymnasium spaces for env spaces
import time
import threading
import logging
import pickle
import glob
import datetime
import uuid
from collections import deque
from flask import Flask, jsonify
import requests

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from dm_robotics.transformations import transformations as tr
from ur5e_sim.controllers.opspace import OpSpaceController
from ur5e_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

class RealRobotInterface:
    def __init__(self, ip):
        self.url = f"http://{ip}:5000"
        self.target_q = None
        self.target_g = None
        self.lock = threading.Lock()
        self.running = True
        # Start sender thread
        self.thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.thread.start()

    def update(self, q, gripper):
        """Update target data, returns immediately."""
        with self.lock:
            self.target_q = q
            self.target_g = gripper

    def _sender_loop(self):
        while self.running:
            q, g = None, None
            with self.lock:
                if self.target_q is not None:
                    q = self.target_q.tolist()
                    g = int(np.clip(self.target_g * 255, 0, 255))

            if q is not None:
                try:
                    # Send servoj command
                    requests.post(f"{self.url}/servoj", json={"q": q}, timeout=0.02)
                    requests.post(f"{self.url}/move_gripper", json={"gripper_pos": g}, timeout=0.02)
                except:
                    pass # Ignore errors to prevent lag

            time.sleep(0.01) # 100Hz

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_ur5e.xml"
_UR5E_HOME = np.asarray([0, -1.57, 1.57, -1.57, -1.57, 0])

_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])
# User requested max obstacles = 64
_MAX_OBSTACLES = 64

class UR5eStackCubeGymEnv(MujocoGymEnv, gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    POTENTIAL_SCALE = 10.0

    # Reward Weights (Sum of base = 1.0, Extras added on top)
    REWARD_WEIGHTS = {
        "reach": 0.65,
        "grasp": 0.05,
        "lift": 0.05,
        "move": 0.25,
        "place_down": 0.5, # Extra
        "release": 0.5,    # Extra
        "grasp_bonus": 10.0,
        "lift_bonus": 5.0,
        "move_bonus": 10.0,
        "full_bonus": 15.0,
    }

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        config = None,
        hz = 10,
        real_robot: bool = False,
        real_robot_ip: str = "127.0.0.1",
        task_stage: str = "full",
        load_state_dir: str = None,
        save_state_dir: str = None,
        success_buffer_size: int = 100,
    ):
        self.hz = hz
        self.real_robot = real_robot
        self.real_robot_ip = real_robot_ip
        self.image_obs = image_obs
        self.task_stage = task_stage
        self.load_state_dir = load_state_dir
        self.save_state_dir = save_state_dir
        self.success_buffer_size = success_buffer_size
        self.success_state_buffer = []
        self.success_history = deque(maxlen=100)

        if config is not None and hasattr(config, 'ACTION_SCALE'):
            self._action_scale = np.array(config.ACTION_SCALE)
        else:
            self._action_scale = action_scale

        MujocoGymEnv.__init__(
            self,
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        print(f"[UR5eEnv] Initialized with _MAX_OBSTACLES={_MAX_OBSTACLES}")

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.env_step = 0
        self.intervened = False
        self._grasp_counter = 0

        # UR5e has 6 joints
        self._ur5e_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 7)]
        )
        self._ur5e_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 7)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]
        self._target_cube_id = self._model.body("target_cube").id
        self._target_cube_geom_id = self._model.geom("target_geom").id
        self._target_cube_z = self._model.geom("target_geom").size[2]

        # Initialize state variables
        self._z_init = 0.0
        self._floor_collision = False

        # Find physical gripper joint for accurate state reading
        # Typically "robot0:2f85:right_driver_joint" for Robotiq 2F-85
        self._gripper_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:2f85:right_driver_joint")

        # Pre-cache pillar IDs for fast collision checking
        # Use SET for O(1) collision checks (critical for performance)
        self._pillar_geom_ids = set()
        self._pillar_info = [] # Cache for _get_obstacle_state: list of (id, type)
        # Search for all pillar geoms up to _MAX_OBSTACLES or until not found
        # Typically XML has limited number, but we scan robustly
        for i in range(1, _MAX_OBSTACLES + 1): # Scan for potential pillars
            id_cyl = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_cyl_{i}")
            if id_cyl != -1:
                self._pillar_geom_ids.add(id_cyl)
                self._pillar_info.append((id_cyl, 'cyl'))

            id_box = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_box_{i}")
            if id_box != -1:
                self._pillar_geom_ids.add(id_box)
                self._pillar_info.append((id_box, 'box'))

        # Cache block ID
        self._block_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "block")

        # Cache floor ID for non-terminal collision checks
        self._floor_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Cache gripper and robot geom IDs
        self._gripper_geom_ids = set()
        self._gripper_pad_geom_ids = set() # Specific for strict grasp detection
        self._left_pad_ids = set()
        self._right_pad_ids = set()
        self._robot_geom_ids = set()
        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body_id = self._model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if name and ("pad" in name or "finger" in name or "2f85" in name):
                self._gripper_geom_ids.add(i)

            # Identify inner pads for strict grasp logic
            if name and "pad" in name:
                self._gripper_pad_geom_ids.add(i)
                if "left" in name:
                    self._left_pad_ids.add(i)
                elif "right" in name:
                    self._right_pad_ids.add(i)

            # Robustly identify robot parts
            if body_name and ("robot0" in body_name or "ur5e" in body_name or "2f85" in body_name):
                self._robot_geom_ids.add(i)

        # Force collision properties for all robot geoms to ensure they interact with pillars
        # Default XML might have them as visual-only (contype=0)
        self._robot_geom_ids_list = list(self._robot_geom_ids)
        for i in self._robot_geom_ids:
            self._model.geom_contype[i] = 1
            self._model.geom_conaffinity[i] = 1
            self._model.geom_solimp[i] = np.array([0.99, 0.999, 0.001, 0.5, 2])
            self._model.geom_solref[i] = np.array([0.005, 1])

        # Initialize Persistent Controller (Zero-Allocation)
        self._opspace_controller = OpSpaceController(self._model, self._ur5e_dof_ids)

        # [NEW] Cache Robot Body IDs for Relative Vector Observation
        self._shoulder_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "robot0:ur5e:upper_arm_link")
        self._elbow_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "robot0:ur5e:forearm_link")

        print(f"[UR5eEnv] Cached {len(self._robot_geom_ids)} Robot Geoms, {len(self._pillar_geom_ids)} Pillar Geoms.")

        # User requested to remove all image observation logic to prevent overhead
        self.observation_space = gymnasium_spaces.Dict(
            {
                "state": gymnasium_spaces.Dict(
                    {
                            "ur5e/tcp_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "ur5e/tcp_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "ur5e/joint_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "ur5e/joint_vel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "ur5e/gripper_pos": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "block_pos_rel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "target_pos_rel": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "cube_grasped": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "obstacle_state": gymnasium_spaces.Box(
                                -np.inf, np.inf, shape=(24,), dtype=np.float32
                            ),
                        }
                    ),
                }
            )
        self.action_space = gymnasium_spaces.Box(
                    low=np.asarray([-1.0, -1.0, -1.0, -1.0]),
                    high=np.asarray([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
        )

        try:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(
                self.model,
                self.data,
            )
            # Optimize: Force lower resolution for 'human' render on WSL to reduce X Server lag
            if self.render_mode == "human":
                if hasattr(self._viewer, 'width'):
                    self._viewer.width = 640 # Reduced from potentially high defaults
                if hasattr(self._viewer, 'height'):
                    self._viewer.height = 480
            else:
                if hasattr(self._viewer, 'width'):
                    self._viewer.width = render_spec.width
                if hasattr(self._viewer, 'height'):
                    self._viewer.height = render_spec.height

            if self.render_mode == "human":
                self._viewer.render(self.render_mode)
        except ImportError:
            print("Warning: Could not initialize MujocoRenderer. Rendering might be disabled.")
            self._viewer = None
        except Exception as e:
             print(f"Warning: Failed to initialize MujocoRenderer: {e}")
             self._viewer = None

        self._safe_geom_ids = set() # Safe for PILLARS (static env)
        safe_names = ["block", "floor", "target_geom", "target"]
        for name in safe_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._safe_geom_ids.add(gid)
            else:
                print(f"Warning: Safe geom '{name}' not found in model.")

        self._robot_safe_geom_ids = set() # Safe for ROBOT
        robot_safe_names = ["block", "target_geom", "target"] # Floor is NOT safe for robot
        for name in robot_safe_names:
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self._robot_safe_geom_ids.add(gid)

        if self.real_robot:
            self._robot_interface = RealRobotInterface(self.real_robot_ip)
            self._start_monitor_server()
            self._connect_real_robot()

    def _connect_real_robot(self):
        url = f"http://{self.real_robot_ip}:5000/clearerr"
        print(f"[Sim] Connecting to Real Robot Server at {url}...")
        try:
            requests.post(url, timeout=2.0)
            print("[Sim] Connected to Real Robot Server.")
        except Exception as e:
            print(f"[Sim] Failed to connect to Real Robot Server: {e}")

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        loaded_state = False
        self.episode_reward = 0.0
        self._prev_potential = 0.0
        self._grasp_counter = 0

        # Try Loading State if configured
        if self.load_state_dir:
            success_files = glob.glob(os.path.join(self.load_state_dir, "success", "*.pkl"))
            if success_files:
                try:
                    chosen_file = np.random.choice(success_files)
                    with open(chosen_file, 'rb') as f:
                        states_list = pickle.load(f)

                    if states_list:
                        state_data = np.random.choice(states_list)
                        self._data.qpos[:] = state_data['qpos']
                        self._data.qvel[:] = state_data['qvel']

                        # Restore Control (Actuators) if present
                        # Crucial for gripper state (Open/Closed)
                        if 'ctrl' in state_data:
                            self._data.ctrl[:] = state_data['ctrl']
                        else:
                            # Backward Compatibility:
                            # If loading old data (no ctrl saved) for stages that require a grasp (lift, move, place),
                            # we MUST force the gripper to be closed. Otherwise default ctrl=0 (Open) drops the object.
                            if self.task_stage in ["lift", "move", "place", "full"]:
                                # Force gripper closed (255)
                                self._data.ctrl[self._gripper_ctrl_id] = 255
                                # print(f"[Sim] Legacy state loaded for '{self.task_stage}'. Forcing gripper CLOSED.")

                        # Restore Target Position if present (New Feature)
                        if 'target_pos' in state_data:
                            self._model.body_pos[self._target_cube_id][:2] = state_data['target_pos']
                        else:
                            # Legacy Default Target
                            self._model.body_pos[self._target_cube_id][:2] = np.array([0.4, 0.25])

                        # Set block pos specifically if needed, but qpos usually covers it
                        # If block is a free joint, it's in qpos.

                        mujoco.mj_forward(self._model, self._data)

                        # Restore Meta-State
                        self.episode_reward = state_data.get('accumulated_reward', 0.0)
                        self._prev_potential = state_data.get('prev_potential', 0.0)
                        self._z_init = state_data.get('z_init', 0.02)
                        self._init_dist_reach = state_data.get('init_dist_reach', 1.0)
                        self._init_dist_move = state_data.get('init_dist_move', 1.0)

                        # Restore Pillar Configuration (Model State) if present
                        # If missing (old data), fall back to randomization to avoid XML defaults
                        pillar_config = state_data.get('pillar_config')

                        block_pos = self._data.sensor("block_pos").data
                        target_pos = self._data.body("target_cube").xpos

                        if pillar_config:
                            # 1. Reset all known pillars to far away first (clean slate)
                            for gid, _ in self._pillar_info:
                                self._model.geom_pos[gid] = np.array([10.0, 10.0, 0.0])

                            # 2. Restore config
                            for gid, params in pillar_config.items():
                                # Check if gid still exists (xml might change, though unlikely)
                                if gid < self._model.ngeom:
                                    self._model.geom_pos[gid] = params['pos']
                                    self._model.geom_size[gid] = params['size']
                                    self._model.geom_rgba[gid] = params['rgba']

                                    # Restore Physics Props
                                    self._model.geom_contype[gid] = params.get('contype', 1)
                                    self._model.geom_conaffinity[gid] = params.get('conaffinity', 1)
                                    self._model.geom_solimp[gid] = params.get('solimp', np.array([0.95, 0.99, 0.001, 0.5, 2]))
                                    self._model.geom_solref[gid] = params.get('solref', np.array([0.005, 1]))
                                    self._model.geom_margin[gid] = params.get('margin', 0.0)

                            # print(f"[Sim] Restored {len(pillar_config)} pillars from state.")
                        else:
                            # Fallback: Randomize to get "2 black pillars" instead of "4 white" defaults
                            # Note: This won't match the grasp state perfectly, but is better than broken defaults.
                            self._randomize_pillars(block_pos[:2], target_pos[:2])
                            print("[Sim] Warning: Old state file detected (no pillar_config). Randomized pillars.")

                        # Sync Mocap to Loaded State
                        # If we don't do this, the mocap remains at XML default (Home),
                        # and the first step will snap the robot back to Home.
                        tcp_pos = self._data.sensor("2f85/pinch_pos").data
                        # Assuming pinch_site orientation is what we want for mocap (usually aligned)
                        # We need to extract quaternion from the site or sensor
                        # Mocap usually controls the pinch site frame.

                        # Note: sensor data for orientation is 4D quat (w,x,y,z) or matrix.
                        # Checking xml, "2f85/pinch_pos" is a 'framepos' sensor.
                        # We might need a 'framequat' sensor or just read the site directly.
                        # Reading site xpos/xquat is safer after mj_forward.

                        site_id = self._model.site("pinch").id
                        self._data.mocap_pos[0] = self._data.site_xpos[site_id].copy()

                        # Convert site matrix to quat for mocap
                        mujoco.mju_mat2Quat(self._data.mocap_quat[0], self._data.site_xmat[site_id])

                        print(f"[Sim] Loaded state from {os.path.basename(chosen_file)}. Inherited Reward: {self.episode_reward}")
                        loaded_state = True
                except Exception as e:
                    print(f"[Sim] Failed to load state: {e}")

        if not loaded_state:
            if self.real_robot:
                try:
                    # Align Sim to Real Robot
                    print("[Sim] Aligning Simulation to Real Robot...")
                    resp = requests.post(f"http://{self.real_robot_ip}:5000/getq", timeout=1.0)
                    real_q = np.array(resp.json()['q'])
                    self._data.qpos[self._ur5e_dof_ids] = real_q
                    print(f"[Sim] Aligned to Q: {real_q}")
                except Exception as e:
                    print(f"[Sim] Failed to align to real robot: {e}")
                    self._data.qpos[self._ur5e_dof_ids] = _UR5E_HOME
            else:
                self._data.qpos[self._ur5e_dof_ids] = _UR5E_HOME

            mujoco.mj_forward(self._model, self._data)

            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            self._data.mocap_pos[0] = tcp_pos

            # Randomize both Block and Target positions independently
            # Enforce minimum distance > 0.3 to prevent overlap/confusion
            while True:
                block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
                target_xy = np.random.uniform(*_SAMPLING_BOUNDS)
                if np.linalg.norm(block_xy - target_xy) > 0.3:
                    break

            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
            self._model.body_pos[self._target_cube_id][:2] = target_xy

            self._randomize_pillars(block_xy, target_xy)

            mujoco.mj_forward(self._model, self._data)

            # Capture actual initial Z of the block for lift reward logic
            self._z_init = self._data.sensor("block_pos").data[2]

            # Initialize distances for potential normalization
            block_pos = self._data.sensor("block_pos").data
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            target_pos = self._data.body("target_cube").xpos

            self._init_dist_reach = np.linalg.norm(block_pos - tcp_pos) + 1e-6
            self._init_dist_move = np.linalg.norm(block_pos - target_pos) + 1e-6

            # Initialize previous potential
            # [Fix] Set initial potential to 0.0 so that the cumulative reward reflects the full potential (starts from 0).
            current_potential, self._latest_potentials = self._calculate_potential(block_pos, tcp_pos, target_pos, False)
            self._prev_potential = 0.0

        self._cached_obstacle_state = self._compute_obstacle_state_once()
        self._z_success = self._z_init + self._target_cube_z * 2

        # Recalculate potential to be safe (if loaded state)
        if loaded_state:
             block_pos = self._data.sensor("block_pos").data
             tcp_pos = self._data.sensor("2f85/pinch_pos").data
             target_pos = self._data.body("target_cube").xpos
             is_grasped = self._check_grasp()
             current_potential, self._latest_potentials = self._calculate_potential(block_pos, tcp_pos, target_pos, is_grasped)
             # Don't reset _prev_potential here, use the loaded one

        self._floor_collision = False
        self.env_step = 0
        self.success_counter = 0
        self._grasp_duration_counter = 0
        self._stage_rewards = {
            "touched": False,
            "lifted": False,
            "hovered": False
        }

        # [New] Track stage completion for full task uniformity
        self._completed_stages = {
            "grasp": False,
            "lift": False,
            "move": False
        }

        # If loading from a curriculum state, mark previous stages as done to avoid double rewards?
        # Or, since 'episode_reward' inherits the value, we should NOT re-award them.
        # But for 'full' task starting from scratch (random reset), these are all False.
        # If we load a 'lift' state, 'grasp' is arguably done.
        # However, checking 'is_grasped' in step() might trigger the reward again if we aren't careful.
        # To be safe: if 'episode_reward' > 0 (inherited), we assume some stages are done.
        # A simple heuristic: check the state conditions immediately.

        if loaded_state:
             # Check what's already done so we don't double count
             is_grasped = self._check_grasp()
             block_pos = self._data.sensor("block_pos").data
             if is_grasped:
                 self._completed_stages["grasp"] = True
             if is_grasped and block_pos[2] > (self._z_init + 0.05):
                 self._completed_stages["lift"] = True
             # Move/Place usually aren't loaded as start states for 'full' unless debugging

        obs = self._compute_observation()

        # Add physical gripper state to info
        info = {"succeed": False}
        if self._gripper_joint_id != -1:
            raw_pos = self._data.qpos[self._gripper_joint_id]
            # Normalize approx 0~0.8 to 0~1
            info["phys_gripper_pos"] = np.clip(raw_pos / 0.8, 0, 1)
        else:
             info["phys_gripper_pos"] = 0.0

        return obs, info
    def _compute_obstacle_state_once(self):
            obs_state = np.zeros((_MAX_OBSTACLES, 7), dtype=np.float32)
            idx = 0
            for gid, ptype in self._pillar_info:
                if idx >= _MAX_OBSTACLES:
                    break

                # 直接讀取 MuJoCo 數據
                pos = self._model.geom_pos[gid]
                size = self._model.geom_size[gid]

                if ptype == 'cyl':
                    obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[0], size[1]]
                else:
                    obs_state[idx] = [1.0, pos[0], pos[1], pos[2], size[0], size[1], size[2]]
                idx += 1
            return obs_state.flatten()
    def _get_obstacle_state(self):
        if not hasattr(self, '_cached_obstacle_state'):
            self._cached_obstacle_state = self._compute_obstacle_state_once()
        return self._cached_obstacle_state

    def _randomize_pillars(self, block_xy, target_xy):
        safe_dist = 0.18 # Increased from 0.14 as requested
        start_pos = np.array([0.3, 0.0])

        def get_safe_pos():
            for _ in range(100):
                px = self._random.uniform(0.2, 0.6)
                py = self._random.uniform(-0.3, 0.3)
                pos = np.array([px, py])
                if (np.linalg.norm(pos - block_xy) > safe_dist and
                    np.linalg.norm(pos - target_xy) > safe_dist and
                    np.linalg.norm(pos - start_pos) > safe_dist):
                    return pos
            return np.array([0.8, 0.8])

        # We assume there are roughly up to 4 pillars in the XML, but loop safely
        # Only ACTIVE pillars (index 1) are randomized. Others moved away.
        for i in range(1, 5):
            name = f"pillar_cyl_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                if i == 1:
                    pos = get_safe_pos()
                else:
                    pos = np.array([10.0, 10.0]) # Move far away

                self._model.geom_pos[body_id][:2] = pos
                radius = self._random.uniform(0.02, 0.03)
                half_height = self._random.uniform(0.0625, 0.1075)
                self._model.geom_size[body_id] = [radius, half_height, 0]
                self._model.geom_pos[body_id][2] = half_height
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]

                self._model.geom_contype[body_id] = 1
                self._model.geom_conaffinity[body_id] = 1
                self._model.geom_solimp[body_id] = np.array([0.95, 0.99, 0.001, 0.5, 2])
                self._model.geom_solref[body_id] = np.array([0.005, 1])
                self._model.geom_margin[body_id] = 0.005

        for i in range(1, 5):
            name = f"pillar_box_{i}"
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if body_id != -1:
                if i == 1:
                    pos = get_safe_pos()
                else:
                    pos = np.array([10.0, 10.0]) # Move far away

                self._model.geom_pos[body_id][:2] = pos
                hx = self._random.uniform(0.02, 0.03)
                hy = self._random.uniform(0.02, 0.03)
                hz = self._random.uniform(0.1025, 0.1675)
                self._model.geom_size[body_id] = [hx, hy, hz]
                self._model.geom_pos[body_id][2] = hz
                self._model.geom_rgba[body_id] = [0.0, 0.0, 0.0, 1.0]

                self._model.geom_contype[body_id] = 1
                self._model.geom_conaffinity[body_id] = 1
                self._model.geom_solimp[body_id] = np.array([0.95, 0.99, 0.001, 0.5, 2])
                self._model.geom_solref[body_id] = np.array([0.005, 1])
                self._model.geom_margin[body_id] = 0.005

    def _start_monitor_server(self):
        import socket
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        try:
            app = Flask("SimMonitor")
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)

            @app.route('/getstate', methods=['POST'])
            def get_state():
                try:
                    pos = self._data.sensor("2f85/pinch_pos").data.tolist()
                except:
                    pos = self._data.site_xpos[self._pinch_site_id].tolist()

                site_mat = self._data.site_xmat[self._pinch_site_id].reshape(9)
                quat_mujoco = np.zeros(4)
                mujoco.mju_mat2Quat(quat_mujoco, site_mat)

                pose = [
                    pos[0], pos[1], pos[2],      # x, y, z
                    quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0] # qx, qy, qz, qw
                ]
                g = self._data.ctrl[self._gripper_ctrl_id] / 255.0
                return jsonify({
                    "pose": pose,
                    "gripper_pos": g,
                    "vel": [0]*6,
                    "force": [0]*3,
                    "torque": [0]*3
                })

            def run_app():
                target_port = 5000
                while is_port_in_use(target_port):
                    print(f"[SimMonitor] Port {target_port} is busy, trying next...")
                    target_port += 1
                    if target_port > 5010:
                        print("[SimMonitor] No available ports found (5000-5010).")
                        return

                print(f"[SimMonitor] Monitor Server started at http://127.0.0.1:{target_port}")
                try:
                    app.run(host='0.0.0.0', port=target_port, debug=False, use_reloader=False)
                except Exception as e:
                    print(f"[SimMonitor] Server crashed: {e}")

            t = threading.Thread(target=run_app)
            t.daemon = True
            t.start()

        except Exception as e:
            print(f"[SimMonitor] Initialization failed: {e}")

    def _check_grasp(self):
        left_contact = False
        right_contact = False

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            if contact.geom1 == self._block_geom_id or contact.geom2 == self._block_geom_id:
                other_id = contact.geom2 if contact.geom1 == self._block_geom_id else contact.geom1

                if other_id in self._left_pad_ids:
                    left_contact = True
                elif other_id in self._right_pad_ids:
                    right_contact = True

        # Check command (must be intended to close)
        gripper_cmd = self._data.ctrl[self._gripper_ctrl_id] / 255.0

        # Check physical state (must not be fully collapsed or fully open)
        # 0.0 = Open, 1.0 = Closed
        # If > 0.95, fingers likely touched each other, not the block.
        # If < 0.05, fingers are fully open.
        current_g_pos = 0.0
        if self._gripper_joint_id != -1:
            raw_pos = self._data.qpos[self._gripper_joint_id]
            current_g_pos = np.clip(raw_pos / 0.8, 0, 1)

        valid_width = (current_g_pos > 0.05) and (current_g_pos < 0.95)

        is_grasping = left_contact and right_contact and valid_width and (gripper_cmd > 0.1)

        # Hysteresis Logic (Fix Flicker)
        if is_grasping:
            self._grasp_counter = 5
            return True
        else:
            if self._grasp_counter > 0:
                self._grasp_counter -= 1
                return True
            else:
                return False

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        start_time = time.time()
        self._floor_collision = False # Reset collision flag

        if len(self._action_scale) == 3:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[2]
        else:
            trans_scale = self._action_scale[0]
            grasp_scale = self._action_scale[1]

        x, y, z, grasp = action

        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * trans_scale

        # Clip to Cartesian Bounds (Safety)
        npos = np.clip(pos + dpos, _CARTESIAN_BOUNDS[0], _CARTESIAN_BOUNDS[1])

        # Enforce minimum Z height to avoid floor penetration
        if npos[2] < 0.02:
             npos[2] = 0.02

        self._data.mocap_pos[0] = npos

        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * grasp_scale
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        # Timing vars
        t_ctrl = 0.0
        t_physics = 0.0

        # Capture collision state during physics steps
        collision_count = 0

        for i in range(self._n_substeps):
            if i % 2 == 0:
                t_c0 = time.time()
                tau = self._opspace_controller(
                    model=self._model,
                    data=self._data,
                    site_id=self._pinch_site_id,
                    pos=self._data.mocap_pos[0],
                    ori=self._data.mocap_quat[0],
                    joint=_UR5E_HOME,
                    gravity_comp=True,
                    pos_gains=(400.0, 400.0, 400.0),
                    damping_ratio=4
                )
                t_ctrl += time.time() - t_c0
            self._data.ctrl[self._ur5e_ctrl_ids] = tau

            t_p0 = time.time()
            mujoco.mj_step(self._model, self._data)
            t_physics += time.time() - t_p0

            # Optimize: Fail fast on collision to prevent solver explosion/lag
            # Note: _check_collision now returns a count, >0 means collision
            c_check = self._check_collision()
            if c_check > 0:
                collision_count = c_check
                break

        obs = self._compute_observation()
        rew = self._compute_reward(collision_count)

        if self.image_obs:
            gripper_key = "gripper_pose"
            gripper_val = obs["state"]["gripper_pose"]
        else:
            gripper_key = "ur5e/gripper_pos"
            gripper_val = obs["state"]["ur5e/gripper_pos"]

        if (action[-1] < -0.5 and gripper_val > 0.9) or (
            action[-1] > 0.5 and gripper_val < 0.9
        ):
            grasp_penalty = -0.02
        else:
            grasp_penalty = 0.0

        self.env_step += 1
        terminated = False
        if self.env_step >= 280:
            terminated = True

        t_poll = 0.0
        t_draw = 0.0

        if self.render_mode == "human" and self._viewer:
            # 0. Explicitly disable VSync to reduce blocking time on WSL X Server (Lazy Init)
            if not getattr(self, '_vsync_set', False):
                if glfw.get_current_context():
                    glfw.swap_interval(0)
                    self._vsync_set = True

            # 1. Always poll events to keep window responsive and prevent queue flooding
            t_p0 = time.time()
            glfw.poll_events()
            t_poll = time.time() - t_p0

            # 2. Throttle rendering to max 20Hz (50ms) to prevent VSync/SwapBuffers blocking the physics loop
            # Initialize last_render_time if not present
            if not hasattr(self, '_last_render_time'):
                self._last_render_time = 0.0

            curr_time = time.time()
            # Dynamic throttling: Limit render FPS to self.hz (18) to match simulation speed
            # and prevent X Server queue flooding.
            target_render_dt = 1.0 / self.hz
            if curr_time - self._last_render_time > target_render_dt:
                t_d0 = time.time()
                self._viewer.render(self.render_mode)
                t_draw = time.time() - t_d0
                self._last_render_time = time.time()

        t_sleep = 0.0
        total_time_ms = (time.time() - start_time) * 1000

        if self.render_mode == "human" or self.real_robot:
            dt = time.time() - start_time
            target_dt = 1.0 / self.hz
            sleep_time = max(0, target_dt - dt)
            if sleep_time > 0:
                t_s0 = time.time()
                time.sleep(sleep_time)
                t_sleep = time.time() - t_s0

        # Log if slow (>100ms)
        final_total = (time.time() - start_time) * 1000
        if final_total > 100:
            logging.warning(
                f"[LAG DETECTED] Total={final_total:.1f}ms | "
                f"Phys={t_physics*1000:.1f}ms | "
                f"Ctrl={t_ctrl*1000:.1f}ms | "
                f"Poll={t_poll*1000:.1f}ms | "
                f"Draw={t_draw*1000:.1f}ms | "
                f"Sleep={t_sleep*1000:.1f}ms"
            )

        if self.real_robot:
            try:
                sim_q = self._data.qpos[self._ur5e_dof_ids]
                sim_g = self._data.ctrl[self._gripper_ctrl_id] / 255.0
                self._robot_interface.update(sim_q, sim_g)
            except Exception as e:
                pass
        
        # collision_count is already computed in the loop above
        info = {
            "succeed": False,
            "grasp_penalty": grasp_penalty,
            "total_episode_reward": self.episode_reward + rew # Expose total for logging
        }

        # Add physical gripper state to info
        if self._gripper_joint_id != -1:
            raw_pos = self._data.qpos[self._gripper_joint_id]
            # Normalize approx 0~0.8 to 0~1
            info["phys_gripper_pos"] = np.clip(raw_pos / 0.8, 0, 1)
        else:
             info["phys_gripper_pos"] = 0.0

        # Terminate on "fatal" collisions (pillars/self), but allow floor touches
        # _check_collision sets _floor_collision flag internally
        # [User Request] Disable collision termination entirely to allow full exploration
        # if collision_count > 0 and not self._floor_collision:
        #     terminated = True
        #     rew = 0.0
        #     success = False
        #     self.success_counter = 0
        #     return obs, rew, terminated, False, info

        # Non-fatal floor penalty (Removed as requested)
        # if self._floor_collision:
        #     rew -= 0.005

        # Check Stage Termination Logic
        stage_success = False

        is_grasped = self._check_grasp()

        if is_grasped:
            self._grasp_duration_counter += 1
        else:
            self._grasp_duration_counter = 0

        block_pos = self._data.sensor("block_pos").data
        target_pos = self._data.body("target_cube").xpos
        dist_target = np.linalg.norm(block_pos[:2] - target_pos[:2])

        if self.task_stage == "grasp":
            if is_grasped and self._grasp_duration_counter >= 20:
                rew += self.REWARD_WEIGHTS["grasp_bonus"]
                stage_success = True
        elif self.task_stage == "lift":
            if is_grasped and block_pos[2] > (self._z_init + 0.05):
                rew += self.REWARD_WEIGHTS["lift_bonus"]
                stage_success = True
        elif self.task_stage == "move":
            if is_grasped and block_pos[2] > (self._z_init + 0.05) and dist_target < 0.05:
                rew += self.REWARD_WEIGHTS["move_bonus"]
                stage_success = True
        elif self.task_stage == "grasp_lift":
            # Check Grasp
            if is_grasped and not self._completed_stages["grasp"]:
                if self._grasp_duration_counter >= 5:
                    self._completed_stages["grasp"] = True
                    rew += self.REWARD_WEIGHTS["grasp_bonus"]

            # Check Lift (Must be grasped)
            if self._completed_stages["grasp"] and not self._completed_stages["lift"]:
                if block_pos[2] > (self._z_init + 0.05):
                    self._completed_stages["lift"] = True
                    rew += self.REWARD_WEIGHTS["lift_bonus"]
                    stage_success = True

        elif self.task_stage == "grasp_lift_move":
            # Check Grasp
            if is_grasped and not self._completed_stages["grasp"]:
                if self._grasp_duration_counter >= 5:
                    self._completed_stages["grasp"] = True
                    rew += self.REWARD_WEIGHTS["grasp_bonus"]

            # Check Lift (Must be grasped)
            if self._completed_stages["grasp"] and not self._completed_stages["lift"]:
                if block_pos[2] > (self._z_init + 0.05):
                    self._completed_stages["lift"] = True
                    rew += self.REWARD_WEIGHTS["lift_bonus"]

            # Check Move (Must be lifted)
            if self._completed_stages["lift"] and not self._completed_stages["move"]:
                if dist_target < 0.05 and block_pos[2] > (self._z_init + 0.02):
                        self._completed_stages["move"] = True
                        rew += self.REWARD_WEIGHTS["move_bonus"]
                        rew += self.REWARD_WEIGHTS["full_bonus"] # Task Complete
                        stage_success = True

        elif self.task_stage == "place" or self.task_stage == "full":
            # For FULL task, we want uniform rewards for intermediate stages
            if self.task_stage == "full":
                # Check Grasp
                if is_grasped and not self._completed_stages["grasp"]:
                    # Require some stability or just instant?
                    # Curriculum used 20 steps, but for intermediate reward instant is usually OK
                    # provided we don't spam it. The flag prevents spam.
                    if self._grasp_duration_counter >= 5: # Small stability check
                        self._completed_stages["grasp"] = True
                        rew += self.REWARD_WEIGHTS["grasp_bonus"]
                        # print("Reward: Grasp Complete")

                # Check Lift (Must be grasped)
                if self._completed_stages["grasp"] and not self._completed_stages["lift"]:
                    if block_pos[2] > (self._z_init + 0.05):
                        self._completed_stages["lift"] = True
                        rew += self.REWARD_WEIGHTS["lift_bonus"]
                        # print("Reward: Lift Complete")

                # Check Move (Must be lifted)
                if self._completed_stages["lift"] and not self._completed_stages["move"]:
                    # Relaxed move check (as requested previously, but sticking to logic):
                    # dist < 0.05 and height maintained
                    if dist_target < 0.05 and block_pos[2] > (self._z_init + 0.02):
                         self._completed_stages["move"] = True
                         rew += self.REWARD_WEIGHTS["move_bonus"]
                         # print("Reward: Move Complete")

            # Original full success logic (Place)
            instant_success = self._compute_success(gripper_val)
            if instant_success:
                self.success_counter += 1
            else:
                self.success_counter = 0
            if self.success_counter >= (1.0 / self.control_dt):
                stage_success = True

        if stage_success:
            # For 'full' or 'place', success means final completion
            if self.task_stage in ["full", "place"]:
                rew += self.REWARD_WEIGHTS["full_bonus"]

            info["succeed"] = True
            terminated = True

        # [Fix] Update episode reward BEFORE saving state
        # This ensures the saved file includes the final success bonus (+20.0)
        self.episode_reward += rew

        if stage_success:
            # Save State Buffer Logic (Now uses correct self.episode_reward)
            self._save_success_state()

        if terminated:
            # Update Success History
            self.success_history.append(1 if info["succeed"] else 0)
            success_rate = sum(self.success_history) / len(self.success_history) * 100.0
            info["success_rate"] = success_rate

            # Breakdown: Scale potentials by POTENTIAL_SCALE (100.0)
            p_reach, p_grasp, p_lift, p_move, p_down, p_release = self._latest_potentials

            # Dynamic max potential calculation for display
            w = self.REWARD_WEIGHTS
            scale = self.POTENTIAL_SCALE

            print(f"\nEpisode Finished.")
            print(f"Total Reward: {self.episode_reward:.2f}")
            print(f"Breakdown (Potential):")
            print(f"  Reach:   {p_reach * scale:.2f} / {w['reach'] * scale:.2f}")
            print(f"  Grasp:   {p_grasp * scale:.2f} / {w['grasp'] * scale:.2f}")
            print(f"  Lift:    {p_lift * scale:.2f} / {w['lift'] * scale:.2f}")
            print(f"  Move:    {p_move * scale:.2f} / {w['move'] * scale:.2f}")
            print(f"  Down:    {p_down * scale:.2f} / {w['place_down'] * scale:.2f}")
            print(f"  Release: {p_release * scale:.2f} / {w['release'] * scale:.2f}")

            success_str = f"Success: {info['succeed']}"
            if info["succeed"]:
                # Total Bonus estimate: Grasp(10)+Lift(5)+Move(10) etc.
                success_str += f" Stage '{self.task_stage}' Complete!"
            success_str += f" Success Rate: {success_rate:.1f}%"
            print(f"  {success_str}")

        return obs, rew, terminated, False, info

    def _check_collision(self):
        if self._data.ncon == 0:
            return 0

        # [Optimization] Limit contact checks to avoid O(N) loop lag
        check_limit = min(self._data.ncon, 50)

        unsafe_collisions = 0

        for i in range(check_limit):
            contact = self._data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            # Explicit check: Robot vs Pillar (Forbidden Region)
            if (g1 in self._robot_geom_ids and g2 in self._pillar_geom_ids) or \
               (g2 in self._robot_geom_ids and g1 in self._pillar_geom_ids):
                print(f"Collision detected: Robot hit Pillar (Forbidden Region)")
                unsafe_collisions += 1
                if unsafe_collisions >= 5: break
                continue

            # 1. Pillar Collisions
            is_g1_pillar = g1 in self._pillar_geom_ids
            is_g2_pillar = g2 in self._pillar_geom_ids

            if is_g1_pillar or is_g2_pillar:
                other_id = g2 if is_g1_pillar else g1
                if other_id not in self._safe_geom_ids:
                    unsafe_collisions += 1
                    if unsafe_collisions >= 5: break
                    continue

            # 2. Robot Collisions (including Floor)
            is_g1_robot = g1 in self._robot_geom_ids
            is_g2_robot = g2 in self._robot_geom_ids

            if is_g1_robot or is_g2_robot:
                other_id = g2 if is_g1_robot else g1

                # Treat FLOOR collision as non-fatal warning (fix "hands up")
                if other_id == self._floor_geom_id:
                    self._floor_collision = True
                    # Even non-fatal, it counts as a 'collision event' for Safety reward
                    unsafe_collisions += 1
                    if unsafe_collisions >= 5: break
                    continue

                # Collision if other is NOT safe (e.g. pillar) AND NOT part of robot (self-collision)
                if other_id not in self._robot_safe_geom_ids and other_id not in self._robot_geom_ids:
                    unsafe_collisions += 1
                    if unsafe_collisions >= 5: break
                    continue

        return unsafe_collisions

    def _save_success_state(self):
        """Buffers and saves successful states for curriculum learning."""

        # Capture Model State (Pillars)
        pillar_config = {}
        for gid, _ in self._pillar_info:
            pillar_config[gid] = {
                'pos': self._model.geom_pos[gid].copy(),
                'size': self._model.geom_size[gid].copy(),
                'rgba': self._model.geom_rgba[gid].copy(),
                'contype': self._model.geom_contype[gid],
                'conaffinity': self._model.geom_conaffinity[gid],
                'solimp': self._model.geom_solimp[gid].copy(),
                'solref': self._model.geom_solref[gid].copy(),
                'margin': self._model.geom_margin[gid]
            }

        state_data = {
            'qpos': self._data.qpos.copy(),
            'qvel': self._data.qvel.copy(),
            'ctrl': self._data.ctrl.copy(), # Save actuator state (Gripper)
            'accumulated_reward': self.episode_reward,
            'prev_potential': self._prev_potential,
            'z_init': self._z_init,
            'init_dist_reach': self._init_dist_reach,
            'init_dist_move': self._init_dist_move,
            'pillar_config': pillar_config,
            'target_pos': self._model.body_pos[self._target_cube_id][:2].copy()
        }
        self.success_state_buffer.append(state_data)

        if len(self.success_state_buffer) >= self.success_buffer_size:
            if self.save_state_dir:
                save_dir = os.path.join(self.save_state_dir, "success")
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                uid = str(uuid.uuid4())[:8]
                filename = f"success_states_{timestamp}_{uid}.pkl"
                filepath = os.path.join(save_dir, filename)

                try:
                    with open(filepath, 'wb') as f:
                        pickle.dump(self.success_state_buffer, f)
                    print(f"[Sim] Saved {len(self.success_state_buffer)} success states to {filepath}")
                    self.success_state_buffer = [] # Clear buffer after save
                except Exception as e:
                    print(f"[Sim] Failed to save states: {e}")
            else:
                 # If no save dir, just keep circular buffer or clear to avoid memory leak
                 if len(self.success_state_buffer) > 500:
                     self.success_state_buffer.pop(0)

    def _compute_success(self, gripper_val):
        block_pos = self._data.sensor("block_pos").data
        target_pos = self._data.body("target_cube").xpos

        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.04

        z_success = block_pos[2] > (target_pos[2] + self._target_cube_z + self._block_z * 0.8)

        gripper_open = gripper_val < 0.1

        block_vel = self._data.jnt("block").qvel[:3]
        is_static = np.linalg.norm(block_vel) < 0.05

        return xy_success and z_success and gripper_open and is_static

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["ur5e/tcp_pos"] = tcp_pos.astype(np.float32)

        # [NEW] Joint Positions and Velocities
        obs["state"]["ur5e/joint_pos"] = self._data.qpos[self._ur5e_dof_ids].astype(np.float32)
        obs["state"]["ur5e/joint_vel"] = self._data.qvel[self._ur5e_dof_ids].astype(np.float32)

        # [NEW] TCP Velocity (Linear + Angular)
        # Compute Jacobian for pinch site
        J = np.zeros((6, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, J[:3], J[3:], self._pinch_site_id)
        # Full qvel (including gripper)
        qvel = self._data.qvel
        tcp_vel = J @ qvel
        obs["state"]["ur5e/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255], dtype=np.float32
        )
        obs["state"]["ur5e/gripper_pos"] = gripper_pos

        target_pos = self._data.body("target_cube").xpos.astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        # Relative positions
        obs["state"]["block_pos_rel"] = (block_pos - tcp_pos).astype(np.float32)
        obs["state"]["target_pos_rel"] = (target_pos - tcp_pos).astype(np.float32)

        # Grasped state (1.0 or 0.0)
        is_grasped = 1.0 if self._check_grasp() else 0.0
        obs["state"]["cube_grasped"] = np.array([is_grasped], dtype=np.float32)

        # [NEW] Obstacle State (Multi-Point Relative Vectors)
        # 1. Gather Key Points
        p_tcp = tcp_pos
        # Use data.xpos for body positions in newer mujoco bindings
        p_elbow = self._data.xpos[self._elbow_body_id]
        p_shoulder = self._data.xpos[self._shoulder_body_id]

        # 2. Find 2 Nearest Obstacles (based on TCP distance)
        active_obstacles = []
        for gid, ptype in self._pillar_info:
            obs_pos = self._model.geom_pos[gid]
            # Skip inactive (far away)
            if obs_pos[0] > 5.0:
                continue

            # Distance to TCP for sorting
            d_tcp = np.linalg.norm(obs_pos - p_tcp)

            # Size
            obs_size = self._model.geom_size[gid]
            # We need 3D size.
            # Cylinder: size=[radius, half_height, 0]
            # Box: size=[hx, hy, hz]
            if ptype == 'cyl':
                s_obs = np.array([obs_size[0], obs_size[0], obs_size[1]], dtype=np.float32)
            else:
                s_obs = np.array([obs_size[0], obs_size[1], obs_size[2]], dtype=np.float32)

            active_obstacles.append({
                'pos': obs_pos,
                'size': s_obs,
                'dist': d_tcp
            })

        # Sort by distance
        active_obstacles.sort(key=lambda x: x['dist'])

        # Take top 2
        nearest = active_obstacles[:2]

        # 3. Construct Feature Vector (24 floats)
        obs_vec = []
        for obs_item in nearest:
            o_pos = obs_item['pos']
            o_size = obs_item['size']

            v_tcp = o_pos - p_tcp
            v_elbow = o_pos - p_elbow
            v_shoulder = o_pos - p_shoulder

            obs_vec.extend(v_tcp)
            obs_vec.extend(v_elbow)
            obs_vec.extend(v_shoulder)
            obs_vec.extend(o_size)

        # Pad if needed
        while len(obs_vec) < 24:
            obs_vec.append(0.0)

        obs["state"]["obstacle_state"] = np.array(obs_vec, dtype=np.float32)

        return obs

    def _is_block_placed(self, block_pos, target_pos):
        xy_dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
        xy_success = xy_dist < 0.04
        # Relax Z threshold to 0.5 to prevent flickering "placed" state
        z_success = block_pos[2] > (target_pos[2] + self._target_cube_z + self._block_z * 0.5)
        return xy_success and z_success

    def _calculate_potential(self, block_pos, tcp_pos, target_pos, is_grasped):
        # 1. Reach Potential
        dist_reach = np.linalg.norm(block_pos - tcp_pos)
        # Normalize: 1.0 when close, 0.0 when at start distance (or further)
        # Using tanh to smoothly saturate
        # Reduce scale from 5.0 to 1.0 to prevent saturation at start (Vanishing Gradient)
        phi_reach = 1 - np.tanh(1.0 * dist_reach / self._init_dist_reach)

        # Determine effective grasp: Real grasp OR Success state (placed)
        is_placed = self._is_block_placed(block_pos, target_pos)

        # New Potentials for Place
        phi_place_down = 0.0
        phi_release = 0.0

        # If placed, we force maximal potentials to represent "Task Complete"
        # However, "is_placed" (legacy) might just mean "Aligned & Above".
        # We refine this for the new rewards: "Down" requires actual proximity.

        target_z_surface = target_pos[2] + self._target_cube_z + self._block_z
        dist_down = max(0.0, block_pos[2] - target_z_surface)

        # Check if actually "Down" (close to surface)
        is_actually_down = is_placed and (dist_down < 0.02)

        if is_placed:
            phi_reach = 1.0
            phi_move = 1.0
            phi_lift = 1.0

            # Refined Down Potential: Even if "is_placed" (aligned), reward lowering.
            # If high up (dist_down large), phi < 1.0. If on surface, phi ~ 1.0.
            # We explicitly calculate it here instead of forcing 1.0 immediately.
            phi_place_down = np.exp(-10.0 * dist_down)

            # Release Potential: Only active if placed AND Down
            if is_actually_down:
                gripper_val = self._data.ctrl[self._gripper_ctrl_id] / 255.0
                phi_release = 1.0 - gripper_val
            else:
                phi_release = 0.0

            effective_grasp = 1.0 # Considered "grasped" in terms of previous stages being done
        else:
            effective_grasp = 1.0 if is_grasped else 0.0
            phi_move = 0.0
            phi_lift = 0.0

            # 2. Lift Potential (only if grasped)
            if effective_grasp > 0.5:
                # Lift target: ~5cm above initial height
                dist_lift = max(0.0, block_pos[2] - self._z_init)
                # Saturate at 5cm using tanh
                phi_lift = np.tanh(60.0 * dist_lift)

                # 3. Move Potential (only if lifted > 2cm)
                if dist_lift > 0.02:
                     dist_move = np.linalg.norm(block_pos - target_pos)
                     phi_move = 1 - np.tanh(1.0 * dist_move / self._init_dist_move)

                     # 4. Place Down Potential (only if moved close to target)
                     # Condition: XY distance < 0.05
                     if dist_move < 0.05:
                         # Z distance to target surface
                         # target_z is target_cube_z + block_z/2 approx?
                         # block_pos[2] target is self._z_init (floor/table) + block size
                         # Actually target surface Z is self._model.geom("target_geom").pos[2] ?? No, body pos.
                         # We want block to be at target_pos[2] + block_z
                         target_z_surface = target_pos[2] + self._target_cube_z + self._block_z
                         dist_down = max(0.0, block_pos[2] - target_z_surface)
                         # We want to minimize dist_down.
                         # Initial hover height is approx 0.05 above that.
                         # Potential = 1.0 when dist_down is 0.
                         phi_place_down = np.exp(-10.0 * dist_down)

        # Mask potentials based on curriculum stage
        if self.task_stage == "grasp":
            phi_lift = 0.0
            phi_move = 0.0
            phi_place_down = 0.0
            phi_release = 0.0
        elif self.task_stage == "lift":
            phi_move = 0.0
            phi_place_down = 0.0
            phi_release = 0.0
        elif self.task_stage == "grasp_lift":
            phi_move = 0.0
            phi_place_down = 0.0
            phi_release = 0.0
        elif self.task_stage == "move":
            # Move implies we stop after moving, so maybe down/release not needed?
            # User request specifically for "place", assuming "full" or "place" stage.
            pass

        # Total Potential (Normalized to 1.0 base + extras)
        w_reach = self.REWARD_WEIGHTS["reach"]
        w_grasp = self.REWARD_WEIGHTS["grasp"]
        w_lift = self.REWARD_WEIGHTS["lift"]
        w_move = self.REWARD_WEIGHTS["move"]
        w_place_down = self.REWARD_WEIGHTS["place_down"]
        w_release = self.REWARD_WEIGHTS["release"]

        # Note: phi_place_down and phi_release are gated effectively by the sequence logic above.
        # phi_place_down is only non-zero if dist_move < 0.05 (and grasped).
        # phi_release is only non-zero if is_placed.

        base_potential = w_reach * phi_reach + effective_grasp * (w_grasp + w_lift * phi_lift + w_move * phi_move)

        # Add Place potentials (Gated by effective grasp for Down, but Release happens when placed)
        # Down requires grasp. Release requires placed (which might technically be ungrasped, but logic handles it).

        extra_potential = effective_grasp * w_place_down * phi_place_down + w_release * phi_release

        potential = base_potential + extra_potential

        return potential, (w_reach * phi_reach, effective_grasp * w_grasp, effective_grasp * w_lift * phi_lift, effective_grasp * w_move * phi_move, effective_grasp * w_place_down * phi_place_down, w_release * phi_release)

    def _compute_reward(self, collision_count) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        target_pos = self._data.body("target_cube").xpos
        is_grasped = self._check_grasp()
        # collision_count unused as Safety reward is removed

        current_potential, potentials = self._calculate_potential(block_pos, tcp_pos, target_pos, is_grasped)
        self._latest_potentials = potentials

        # Step reward is difference in potential
        step_rew = (current_potential - self._prev_potential) * self.POTENTIAL_SCALE

        # Update previous potential
        self._prev_potential = current_potential

        return step_rew