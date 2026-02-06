# 其中mask 都要转成二值图（0, 1） 进行训练 并且训练集和验证集都是封闭 填充的区域，不是只包含边界的区域

python pre_tongue_fast.py -i "./data/train/Tongueset3/img" -gt "./data/train/Tongueset3/gt" -o "./data/train_npz" --img_name_suffix ".jpg" --label_id 1 --data_name "tongueset3" --device "cuda:0" --batch_size 4 
# image -> npz  训练集 进行增强
python pre_tongue_fast.py -i "./data/train/TongeImageDataset/extracted/dataset" -gt "./data/train/TongeImageDataset/extracted/groundtruth/mask" -o "./data/test_npz" --img_name_suffix ".bmp" --label_id 1 --data_name "tongueset2" --device "cuda:0" --batch_size 4 --no_augment


# image -> npz  验证集 不进行增强


python train.py
# npz -> pth

python predict.py
# 推理部分
