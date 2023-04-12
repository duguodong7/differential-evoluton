CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
                                        main.py -b 256 \
										--workers 16 \
										--val_workers 16 \
										-vb 3 \
										--opt 'sgd' \
										--epochs 10 \
										--de_epochs 5000 \
										--de_batch_size 128 \
										--mini_batch_size 128 \
										--popsize 20 \
										--de_iters 10 \
										--decay_epochs 30 \
										--model 'resnet50' \
										--sync_bn \
										--lr 0.001 \
										--amp \
										--num-gpu 1 \
                                        --pin_mem True \
                                        --data /root/imagenet \
                                        --log_dir '/root/declc_guodong/log_out/resnet50_3.txt' \
										--pop_init_dir '/root/declc_guodong/resnet50-76.30' 


                                        # --data_wds /root/imagenet_wds/shards \
#10240, 30720 * 4 = 122880 
# 768 * 160 = 122880 
# /root/declc/log_out/hahaha.txt

#74700,  900
#76800,  768.   9000, 900