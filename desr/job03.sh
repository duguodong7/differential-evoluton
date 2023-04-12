#!/bin/bash
#SBATCH -J edsr              # 作业名是 test
#SBATCH -p defq              # 提交到 默认的defq 队列
#SBATCH -N 1                 # 使用2个节点
#SBATCH --ntasks-per-node=1  # 每个节点开启6个进程
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=12    # 每个进程占用一个 cpu 核心
#SBATCH -w node03
#SBATCH --mem=100G           #申请100G内存

#SBATCH --gres=gpu:1         # 如果是gpu任务需要在此行定义gpu数量,此处为1

# module load anaconda

#python main.py --model RDN --scale 2 --patch_size 64 --save rdn2_baseline_x2 --reset --dir_data /home/guodong  \
										#SBATCH -o /home/guodong/EDSR/DE3_pop_out_log.txt             
python main.py --model EDSR  \
				 --scale 2  \
				 --patch_size 96  \
				 --batch_size 16  \
				 --de_epochs 2000 \
				 --de_iters 3 \
				 --de_patch_size 480 \
				 --de_batch_size 320 \
				 --mini_batch_size 16 \
				 --de_test_every 800 \
				 --save haha \
				 --reset \
				 --popsize 6 \
				 --dir_data /home/guodong   \
				 --output_dir /home/guodong/desr \
				 --pop_init_dir  /home/guodong/desr/pretrained/EDSR_ckpt \



				 # --pre_train /home/guodong/EDSR/pretrained/model_1000_bs32.pt \

				 # --pretrain_epoch 20 \
# --no_augment sade_train_ssim_pop20_bs80_ps480_pre1000bs32\