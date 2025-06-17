# python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py \
# --diff_steps 2000 \
# --lr 0.0001 \
# --learning_steps 50000 \
# --save_interval 10000 \
# --seed 102 \
# --noise_schedule sqrt \
# --hidden_dim 128 \
# --bsz 2048 \
# --dataset hellaswag \
# --data_dir /content/DiffuSeq/datasets/hellaswag \
# --vocab bert \
# --seq_len 128 \
# --schedule_sampler lossaware \
# --notes test-qqp
python run_train.py \
--diff_steps 200 \
--lr 5e-5 \
--learning_steps 4000 \
--save_interval 1000 \
--seed 42 \
--noise_schedule linear \
--hidden_dim 128 \
--bsz 64 \
--dataset hellaswag \
--data_dir /content/DiffuSeq/datasets/hellaswag \
--vocab bert \
--seq_len 128 \
--schedule_sampler uniform \
--notes test-hellaswag
