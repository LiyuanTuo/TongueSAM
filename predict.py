from PIL import ImageDraw, Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
from skimage import io
from  utils_metrics import *
from skimage import transform, io, segmentation
from segment.yolox import YOLOX
import random
import warnings
import time

# 永久性地忽略指定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
#########################################################################################################
ts_img_path = './data/test_in/'
model_type = 'vit_b'
checkpoint = './logs/best.pth'  # 使用训练好的模型
# checkpoint = './pretrained_model/sam.pth'  # 使用预训练模型
device = 'cuda:0'
path_out='./data/test_out/'
# segment=YOLOX()  # 推理阶段不使用YOLOX，注释掉以释放GPU显存
ENCODER_BATCH_SIZE = 8  # image_encoder 批量推理的 batch size，显存不够就改小
##############################################################################################################
def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''    
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))
best_iou=0
test_names = sorted(os.listdir(ts_img_path))

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.eval()
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

val_gts=[]
val_preds=[]

t_start = time.time()

# ============ 第一阶段：预处理所有图片 & 批量 image_encoder 推理 ============
all_filenames = []
all_image_data_pre = []   # 预处理后的 uint8 图
all_input_tensors = []     # 送入 encoder 的 tensor
all_tongue_masks = []      # tongue mask tensor

print(f"[阶段1] 预处理 {len(test_names)} 张图片...")
for f in tqdm(test_names, desc="预处理"):
    try:
        pil_image = Image.open(join(ts_img_path, f))
        pil_image = ImageOps.exif_transpose(pil_image)
        image_data = np.array(pil_image)
    except Exception as e:
        print(f"Error loading image with PIL: {e}, falling back to skimage")
        image_data = np.array(Image.open(join(ts_img_path, f)))

    # 使用 cv2.resize 替代 skimage.transform.resize（快 100x+）
    image_data = cv2.resize(image_data, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float64)

    if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
        image_data = image_data[:, :, :3]
    if len(image_data.shape) == 2:
        image_data = np.repeat(image_data[:, :, None], 3, axis=-1)

    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
    image_data_pre[image_data == 0] = 0
    image_data_pre = np.uint8(image_data_pre)

    tongue_mask = np.array(ImageOps.exif_transpose(Image.open("../舌苔抑郁症/mask/" + os.path.splitext(f)[0] + ".png")))

    tongue_mask = cv2.resize(tongue_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # 调试：验证 mask 的实际值分布（只打印第一张）
    if f == test_names[0]:
        print(f"[DEBUG] tongue_mask shape={tongue_mask.shape}, dtype={tongue_mask.dtype}")
        print(f"[DEBUG] tongue_mask unique values: {np.unique(tongue_mask)}")
        print(f"[DEBUG] tongue_mask==0 的像素数: {(tongue_mask==0).sum()}, 非零像素数: {(tongue_mask!=0).sum()}")

    all_image_data_pre.append(image_data_pre.copy())  # 后续可视化用原始预处理图

    image_data_pre[tongue_mask == 0] = [0, 0, 0] # 将非舌头区域像素值设为0，突出舌头部分

    tongue_mask = (tongue_mask > 0).astype(np.float32)

    resize_img = sam_transform.apply_image(image_data_pre)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1))
    input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :].to(device)).cpu()  # 预处理后移回 CPU，避免 673 张占爆显存

    tongue_mask_256 = cv2.resize(tongue_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_torch = torch.as_tensor(tongue_mask_256[None, None, :, :], dtype=torch.float)  # 存 CPU 上

    all_filenames.append(f)
    all_input_tensors.append(input_image)
    all_tongue_masks.append(mask_torch)

# 批量通过 image_encoder（最耗时的部分）  这一部分被AI修改了， 酌情保留
all_embeddings = []
print(f"[阶段2] 批量 image_encoder 推理 (batch_size={ENCODER_BATCH_SIZE}, FP16)...")
with torch.no_grad(), torch.cuda.amp.autocast():  # FP16 半精度：速度翻倍 + 显存减半
    for i in tqdm(range(0, len(all_input_tensors), ENCODER_BATCH_SIZE), desc="Encoder"):
        batch = torch.cat(all_input_tensors[i:i+ENCODER_BATCH_SIZE], dim=0).to(device)
        embeddings = sam_model.image_encoder(batch)
        # 拆分回单张存储，放 CPU 上避免 673 个 embedding 撑爆显存
        for j in range(embeddings.shape[0]):
            all_embeddings.append(embeddings[j:j+1].float().cpu())  # 转回 FP32 存 CPU
        del batch, embeddings
        torch.cuda.empty_cache()

# 释放 input_tensors，节省显存
del all_input_tensors
torch.cuda.empty_cache()

# ============ 第二阶段：逐张 prompt_encoder + mask_decoder + 可视化保存 ============
print(f"[阶段3] Decoder + 保存结果...")
with torch.no_grad():
    for idx in tqdm(range(len(all_filenames)), desc="Decoder+保存"):
        f = all_filenames[idx]
        ts_img_embedding = all_embeddings[idx]
        mask_torch = all_tongue_masks[idx]
        img = all_image_data_pre[idx]

        boxes = None

        if boxes is not None:
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(boxes, (1024,1024))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        else:
            box_torch = None

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=mask_torch.to(device),
        )

        # 使用Mask_Decoder生成分割结果
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=ts_img_embedding.to(device),
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0).astype(np.uint8)  # logit > 0 等价于 sigmoid(logit) > 0.5

        medsam_seg = cv2.resize(medsam_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        ####################################
        medsam_seg[medsam_seg > 0] = 255
        medsam_img = Image.fromarray(medsam_seg)
        medsam_img.save(os.path.join(path_out, os.path.splitext(f)[0] + '.png'))

        ####################################
        # 后面这一部分代码就是为了在原图上叠加边界框和分割结果的可视化，输出到test_out文件夹中
        pred = cv2.Canny(medsam_seg, 100, 200)

        # 向量化边缘绘制（替代逐像素 Python for 循环，速度提升 100x+）
        edge_ys, edge_xs = np.where(pred != 0)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ys = np.clip(edge_ys + dy, 0, 1023)
                xs = np.clip(edge_xs + dx, 0, 1023)
                img[ys, xs, :] = [0, 0, 255]

        image1 = Image.fromarray(medsam_seg)
        image2 = Image.fromarray(img)

        image1 = image1.convert("RGBA")
        image2 = image2.convert("RGBA")
        data1 = image1.getdata()

        new_image = Image.new("RGBA", image2.size)
        new_data = [(0, 0, 128, 96) if pixel1[0] != 0 else (0, 0, 0, 0) for pixel1 in data1]

        new_image.putdata(new_data)
        if boxes is not None:
            draw = ImageDraw.Draw(image2)
            draw.rectangle([boxes[0],boxes[1],boxes[2],boxes[3]],fill=None, outline=(0, 255, 0), width=5)
        image2.paste(new_image, (0, 0), mask=new_image)
        image2.save(os.path.join(path_out, "vis_" + os.path.splitext(f)[0] + '.png'))

t_end = time.time()
print(f"\n总耗时: {t_end - t_start:.1f}s, 平均: {(t_end - t_start) / max(len(all_filenames), 1):.2f}s/张")
