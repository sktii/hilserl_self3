export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.25 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=run0130_1 \
    --actor \
    --render_mode=None \
    --model_part=grasp