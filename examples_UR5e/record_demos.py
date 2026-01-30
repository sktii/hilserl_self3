# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import sys
sys.path.insert(0, '../../../')
import os

# Fix for WSL/Lag: Limit NumPy/OpenMP threading to prevent explosion during opspace control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Fix for WSL/Lag: Force driver-level VSync off to bypass Windows DWM/WSLg composition lag
os.environ["__GL_SYNC_TO_VBLANK"] = "0" # NVIDIA
os.environ["vblank_mode"] = "0"          # Mesa/SW

# Fix for WSL/Lag: Unset MUJOCO_GL=egl if detected, to allow windowed rendering (GLFW)
if os.environ.get("MUJOCO_GL") == "egl":
    print("Pre-emptive fix: Unsetting MUJOCO_GL=egl to allow windowed rendering in record_demos.py")
    del os.environ["MUJOCO_GL"]

# Force JAX to use CPU to avoid GPU/GLFW conflicts in WSL
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Prevent JAX from hogging GPU memory, allowing MuJoCo EGL to run smoothly
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import gc

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_string("model_part", "full", "Curriculum stage to record (grasp, lift, move, place, full).")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints (used to infer curriculum load path).")
flags.DEFINE_string("load_state_path", None, "Path to load curriculum states from (override auto-inference).")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Curriculum Path Logic
    STAGES = ["grasp", "lift", "move", "place", "full", "grasp_lift", "grasp_lift_move"]
    load_state_dir = FLAGS.load_state_path

    # Define save_state_dir to allow chaining demos (saving success states for next task)
    # Default to saving in demo_data/{model_part} which creates a 'success' subdir
    save_state_dir = f"./demo_data/{FLAGS.model_part}"

    STAGES_WITH_PRIOR = ["lift", "move", "place"]

    # 1. Try explicit checkpoint path
    if FLAGS.checkpoint_path:
        if not load_state_dir and FLAGS.model_part in STAGES_WITH_PRIOR:
            idx = STAGES.index(FLAGS.model_part)
            if idx > 0:
                prev_stage = STAGES[idx-1]
                load_state_dir = os.path.join(FLAGS.checkpoint_path, "curriculum", prev_stage)
                print(f"Auto-inferred curriculum load path from checkpoint: {load_state_dir}")

    # 2. Try inferring from demo_data (Pure Demo Workflow) if no checkpoint path
    if not load_state_dir and FLAGS.model_part in STAGES_WITH_PRIOR:
        idx = STAGES.index(FLAGS.model_part)
        if idx > 0:
            prev_stage = STAGES[idx-1]
            demo_prev_dir = f"./demo_data/{prev_stage}"
            # Check if success states exist there
            if os.path.exists(os.path.join(demo_prev_dir, "success")):
                load_state_dir = demo_prev_dir
                print(f"Auto-inferred curriculum load path from demo data: {load_state_dir}")

    # [Fix] Raise error if curriculum stage requires previous state but no path found
    if FLAGS.model_part in ["lift", "move", "place"] and not load_state_dir:
        raise ValueError(
            f"Model part '{FLAGS.model_part}' requires a starting state from the previous stage, "
            "but no source was found.\n"
            "Please provide --checkpoint_path (if you have trained models) OR ensure you have recorded "
            f"successful demos for the previous stage ('{STAGES[STAGES.index(FLAGS.model_part)-1]}') which generated state files in demo_data."
        )

    # Disable classifier to rely on simulation ground truth success for demo recording
    # Use success_buffer_size=1 to ensure even small demo batches (e.g. 2) save states immediately
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        model_part=FLAGS.model_part,
        load_state_dir=load_state_dir,
        save_state_dir=save_state_dir,
        success_buffer_size=1
    )
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    step_count = 0

    # Disable automatic GC to prevent stuttering/accumulation during episode
    # NOTE: User reported "reset to flow running" behavior, which implies GC pauses might be the "flow".
    # But "accumulating" implies memory growth.
    # We keep gc.disable() because it is best practice for high-freq loops.
    # gc.disable()

    # Recursive function to force deep copy of numpy arrays (detaching from MuJoCo views)
    def force_copy(obj):
        if isinstance(obj, np.ndarray):
            return obj.copy()
        elif isinstance(obj, dict):
            return {k: force_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [force_copy(v) for v in obj]
        return obj

    while success_count < success_needed:
        step_count += 1
        actions = np.zeros(env.action_space.sample().shape) 

        next_obs, rew, done, truncated, info = env.step(actions)

        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # [Fix] Periodic GC to prevent memory fragmentation on long runs (WSL specific)
        if step_count % 500 == 0:
            gc.collect()

        # Use force_copy to ensure we don't hold references to MuJoCo memory views
        transition = {
            "observations": force_copy(obs),
            "actions": force_copy(actions),
            "next_observations": force_copy(next_obs),
            "rewards": rew,
            "masks": 1.0 - done,
            "dones": done,
            "infos": force_copy(info),
        }

        trajectory.append(transition)
        
        if step_count % 20 == 0:
            # Use total reward from info if available (handles inheritance), else local return
            display_return = info.get("total_episode_reward", returns)
            pbar.set_description(f"Total Reward: {display_return:.2f}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    # Trajectory items are already copied, just append
                    transitions.append(transition)
                success_count += 1
                pbar.update(1)

            # Explicitly clear trajectory to free memory
            del trajectory[:]
            trajectory = []
            returns = 0

            # Manually run GC between episodes when it's safe
            gc.collect()

            obs, info = env.reset()
            
    # Save to curriculum-specific subdirectory
    save_dir = f"./demo_data/{FLAGS.model_part}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_dir}/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)