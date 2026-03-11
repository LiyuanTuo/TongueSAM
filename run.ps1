# 其中mask 都要转成二值图（0, 1） 进行训练 并且训练集和验证集都是封闭 填充的区域，不是只包含边界的区域

python pre_tongue_fast.py -i "./data/train/Tongueset3/img" -gt "./data/train/Tongueset3/gt" -o "./data/train_npz" --label_id 1 --data_name "tongueset3" --device "cuda:0" --batch_size 4 
# image -> npz  训练集 进行增强
python pre_tongue_fast.py -i "./data/train/TongeImageDataset/extracted/dataset" -gt "./data/train/TongeImageDataset/extracted/groundtruth/mask" -o "./data/test_npz" --label_id 1 --data_name "tongueset2" --device "cuda:0" --batch_size 4 --no_augment

# image -> npz  验证集 不进行增强
# 验证集中有8位的mask， 有1 位的mask

# mask 有逆天数据 mask 与真实值反了
    


python train.py
# npz -> pth

# 原repo中的yolo.pth 是经过微调过的， 类别只有一个就是 tongue ， 而不是coco中的80类别，因为用
# model = YoloBody(num_classes=80, phi='s')
# state = torch.load('segment/yolox.pth', map_location='cpu') size mismatch for head.cls_preds.0.weight: copying a param with shape torch.Size([1, 128, 1, 1]) from checkpoint, the shape in current model is torch.Size([80, 128, 1, 1]).
python predict.py

# 推理部分
