CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
                                        main.py -b 100 \
                                        --config imagenet.yml \
										--workers 8 \
										--val_workers 8 \
										-vb 1 \
										--opt adamw \
										--epochs 10 \
										--de_epochs 5000 \
										--de_batch_size 8000 \
										--mini_batch_size 100 \
										--popsize 6 \
										--de_iters 3 \
										--decay_epochs 30 \
										--model 'vit_snn' \
										--sync_bn \
										--amp \
										--num-gpu 1 \
                                        --pin_mem True \
                                        --data /root/imagenet \
                                        --log_dir '/root/desnn/log_out/vit_snn_0.txt' \
										--pop_init_dir '/root/desnn/top10-checkpoints' \


                                        # --data_wds /root/imagenet_wds/shards \
#10240, 30720 * 4 = 122880 
# 768 * 160 = 122880 
# /root/declc/log_out/hahaha.txt

#74700,  900
#76800,  768.   9000, 900