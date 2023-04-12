训练时
python -m torch.distributed.launch --nproc_per_node=8  train_vit_snn.py
各文件作用：
imagenet.yml 设置超参数
vit_snn 模型
train_vit_snn 训练文件
neuron 从spikjelly扒的文件，因为当时云脑有bug
