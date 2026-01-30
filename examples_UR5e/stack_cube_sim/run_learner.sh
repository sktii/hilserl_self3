export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.25 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=run0130_1 \
    --demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/full/stack_cube_sim_20_demos_2026-01-29_16-01-55.pkl \
    --learner \
    --model_part=full

    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp/stack_cube_sim_20_demos_2026-01-28_13-22-31.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/lift/stack_cube_sim_20_demos_2026-01-28_13-49-33.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/move/stack_cube_sim_20_demos_2026-01-28_14-08-59.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/place/stack_cube_sim_20_demos_2026-01-28_09-15-46.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/full/stack_cube_sim_20_demos_2026-01-29_08-40-54.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/place/stack_cube_sim_20_demos_2026-01-28_09-15-46.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp_lift/stack_cube_sim_20_demos_2026-01-29_10-01-41.pkl \
    #--demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp_lift_move/stack_cube_sim_20_demos_2026-01-29_11-25-26.pkl \

    # --demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp/stack_cube_sim_20_demos_2026-01-28_13-22-31.pkl \
    # --demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp_lift/stack_cube_sim_20_demos_2026-01-29_10-01-41.pkl \
    # --demo_path=/home/wayne/hilserl_selflearning3/examples_UR5e/experiments/stack_cube_sim/demo_data/grasp_lift_move/stack_cube_sim_20_demos_2026-01-29_11-25-26.pkl \
