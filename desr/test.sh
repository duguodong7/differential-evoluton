#!/bin/bash
#SBATCH -J edsr_test              # 作业名是 test
#SBATCH -p defq              # 提交到 默认的defq 队列
#SBATCH -N 1                 # 使用2个节点
#SBATCH --ntasks-per-node=12  # 每个节点开启6个进程
#SBATCH -t 10-00:00:00
#SBATCH --cpus-per-task=1    # 每个进程占用一个 cpu 核心
#SBATCH -w node04
#SBATCH --mem=100G           #申请100G内存

#SBATCH --gres=gpu:1         # 如果是gpu任务需要在此行定义gpu数量,此处为1

# module load anaconda

#python main.py --model RDN --scale 2 --patch_size 64 --save rdn2_baseline_x2 --reset --dir_data /home/guodong  \
										#SBATCH -o /home/guodong/EDSR/DE3_pop_out_log.txt             

python main.py --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K \
				--data_range 801-900 \
				--scale 2 \
				--pre_train /home/guodong/EDSR/pretrained/model_ssim_best.pt \
				--test_only \
				#--self_ensemble


# model_ssim_best.pt
# python main.py --model EDSR  \
# 				 --scale 2  \
# 				 --patch_size 96  \
# 				 --batch_size 32  \
# 				 --de_patch_size 480 \
# 				 --de_batch_size 10 \
# 				 --de_test_every 800 \
# 				 --save de_train_pop20_bs10_ps480_pre1000bs32_ \
# 				 --reset \
# 				 --popsize 20 \
# 				 --pretrain_epoch 30\
# 				 --pre_train /home/guodong/EDSR/pretrained/model_1000_bs32.pt \
# 				 --dir_data /home/guodong               # 执行我编译的的程序


# --no_augment \