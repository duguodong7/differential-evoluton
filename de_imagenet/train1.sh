# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port='29502' \
#                                         main1.py -b 64 \
# 										--workers 16 \
# 										-vb 5 \
# 										--opt 'sgd' \
# 										--epochs 200 \
# 										--de_epochs 0 \
# 										--de_batch_size 32 \
# 										--mini_batch_size 16 \
# 										--de_iters 3 \
# 										--decay_epochs 30 \
# 										--pretrained \
# 										--sync_bn \
# 										--lr 0.001 \
# 										--amp \
# 										--num-gpu 1 \
#                                         --pin_mem True \
#                                         --data /root/imagenet 
  

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29503'\
# 								main1.py -b 128 \
# 										--workers 4 \
# 										-vb 6 \
# 										--opt 'sgd' \
# 										--epochs 300 \
# 										--model 'resnet50' \
# 										--model_load_path '/root/declc_guodong/pretrained/resnet50-0676ba61.pth' \
# 										--de_epochs 0 \
# 										--de_batch_size 76800 \
# 										--mini_batch_size 768 \
# 										--popsize 10 \
# 										--de_iters 5 \
# 										--decay_epochs 30 \
# 										--sync_bn \
# 										--lr 0.001 \
# 										--amp \
# 										--num-gpu 1 \
# 										--data /root/imagenet


CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29504'\
								main1.py -b 64 \
										--workers 4 \
										-vb 12 \
										--opt 'sgd' \
										--epochs 300 \
										--model 'resnet50' \
										--model_load_path '/root/declc_guodong/pretrained/resnet50-0676ba61.pth' \
										--de_epochs 0 \
										--de_batch_size 76800 \
										--mini_batch_size 768 \
										--popsize 10 \
										--de_iters 5 \
										--decay_epochs 30 \
										--sync_bn \
										--lr 0.001 \
										--amp \
										--num-gpu 1 \
										--data /root/imagenet

										# --only_test \
# --log_dir /root/declc/log_out/pretrained_bs32.txt 



#10240