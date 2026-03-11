#%% import packages - Optimized version with batch processing
import numpy as np
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import cv2
from PIL import Image, ImageOps

# set up the parser
parser = argparse.ArgumentParser(description='preprocess grey and RGB images (fast version)')
parser.add_argument('-i', '--img_path', type=str, required=True, help='path to the images')
parser.add_argument('-gt', '--gt_path', type=str, required=True, help='path to the ground truth (gt)')
parser.add_argument('-o', '--npz_path', type=str, required=True, help='path to save the npz files')
parser.add_argument('--data_name', type=str, default='tongue', help='dataset name')
parser.add_argument('--image_size', type=int, default=1024, help='image size')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
# parser.add_argument('--checkpoint', type=str, default='./pretrained_model/sam.pth', help='checkpoint')
parser.add_argument('--checkpoint', type=str, default='./log/best.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for SAM encoder')
parser.add_argument('--no_augment', action='store_true', help='disable data augmentation for faster processing')
args = parser.parse_args()


def simple_preprocess(image, gt, image_size):
    """Simple preprocessing without heavy augmentation"""
    # Resize image and gt to target size
    image = cv2.resize(image, (image_size, image_size))
    gt = cv2.resize(gt, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image, gt


def rotate_image_and_mask(image, mask, angle):
    """Rotate image and mask with the same affine transform."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rot = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_rot = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_rot, mask_rot

def add_random_shadow(image_np):
    """Add a random realistic soft shadow to the image."""
    h, w = image_np.shape[:2]
    num_vertices = np.random.randint(3, 6)
    # 让多边形顶点可以超出图像边界，模仿从界外投射的阴影
    pts = np.random.randint(-w//2, w + w//2, size=(num_vertices, 2)).astype(np.int32)
    
    shadow_canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shadow_canvas, [pts], 255)
    
    # 极大的高斯模糊核让阴影边缘变得非常柔和(模拟环境光的软投影)
    shadow_mask = cv2.GaussianBlur(shadow_canvas, (101, 101), 0)
    
    # 决定阴影的暗度 (0.4 到 0.7之间，越低越暗)
    intensity = np.random.uniform(0.4, 0.7)
    shadow_mask_float = shadow_mask.astype(np.float32) / 255.0
    
    # 1.0表示不受影响，计算后形成遮罩平滑过渡
    light_mask = 1.0 - (shadow_mask_float * (1.0 - intensity))
    light_mask_3d = np.repeat(light_mask[:, :, np.newaxis], image_np.shape[2], axis=2)
    
    # 叠加带有透明度的遮罩
    shadowed_image = image_np.astype(np.float32) * light_mask_3d
    return np.clip(shadowed_image, 0, 255).astype(np.uint8)


def process_single_image(img_path, gt_path, image_name, gt_name, image_size, no_augment=True):
    """Process a single image and return preprocessed data"""

    
     # gt name should be the same as image name  gtdata 是二维的
    gt_data = Image.open(join(gt_path, gt_name))
    gt_data = ImageOps.exif_transpose(gt_data)
    gt_data = np.array(gt_data)


    image_data = Image.open(join(img_path, image_name))
    image_data = ImageOps.exif_transpose(image_data)  
    image_data = np.array(image_data)

    if gt_data.ndim == 3:
        gt_data = gt_data[:, :, 0]  # 取第一个通道作为gt数据，假设gt是单通道图像
    
    if no_augment:
        image_data, gt_data = simple_preprocess(image_data, gt_data, image_size)
    else:
        # Augmentation (rotation-focused)

        # Random rotation (same angle for image and mask)
        # angle = np.random.uniform(-10, 10)
        # image_data, gt_data = rotate_image_and_mask(image_data, gt_data, angle)

        #  TODO add real world shadow augmentation
        image_data, gt_data = simple_preprocess(image_data, gt_data, image_size)

        image_data = add_random_shadow(image_data)


    # 现在tonguemask 的大小一样了
    tonguemask = (gt_data <= 128).astype(np.uint8)  # 二值化，舌头部分为1，其他部分为0
    

    # Binarize gt   128 是舌苔 255 是背景 0 是舌体 但是我们只需要二值化的结果，所以把128的部分设置为1，其他部分设置为0
    gt_data = (gt_data == 128).astype(np.uint8) # fixed this serious bug
    # print(f"Post-Binarization unique: {np.unique(gt_data)}") 最大最小值应该是0和255 但只会有0和1 因为上面已经做了二值化了
    
    # Ensure image is RGB
    if len(image_data.shape) == 2:
        image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
    if image_data.shape[-1] > 3:
        image_data = image_data[:, :, :3]
    
    # Normalize image
    image_data = image_data.astype(np.float32)
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data = np.clip(image_data, lower_bound, upper_bound)
    if image_data.max() > image_data.min():
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255.0
    image_data = np.uint8(image_data)
    
    image_data[tonguemask == 0] = [0, 0, 0]  # 将非舌头区域像素值设为0，突出舌头部分  所以训练的时候就只训练了舌头部分，推理的时候自然也只推理舌头部分

    # Get bounding box from gt
    y_indices, x_indices = np.where(gt_data > 0)
    if len(x_indices) > 0:
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        box = np.array([xmin, ymin, xmax, ymax])
    else:
        box = np.array([0, 0, image_size-1, image_size-1])
    
    return image_data, gt_data, box, tonguemask


def deal_fast(img_path, gt_path, sam_model, sam_transform):
    """Optimized processing with batch encoding"""
    names = sorted(os.listdir(img_path))
    save_path = args.npz_path
    os.makedirs(save_path, exist_ok=True)
    print(f'Processing {len(names)} images...')
    
    imgs = []
    gts = []
    boxes = []
    img_embeddings = []
    tonguemasks = []

    # Collect all preprocessed images first
    batch_images = []
    batch_indices = []
    
    for idx, image_name in enumerate(tqdm(names, desc="Loading images")):
        gt_name = image_name

        try:
            image_data, gt_data, box, tonguemask = process_single_image(
                img_path, gt_path, image_name, gt_name, 
                args.image_size, args.no_augment
            )
            imgs.append(image_data)
            gts.append(gt_data)
            boxes.append(box)
            tonguemasks.append(tonguemask)
            batch_images.append(image_data)
            batch_indices.append(idx)
        except Exception as e:
            print(f"Error processing {gt_name}: {e}")
            continue
    
    # Batch encode images through SAM
    print(f"Encoding {len(batch_images)} images through SAM (batch_size={args.batch_size})...")
    
    for i in tqdm(range(0, len(batch_images), args.batch_size), desc="SAM encoding"):
        batch = batch_images[i:i+args.batch_size]
        
        # Prepare batch tensor
        batch_tensors = []
        for img in batch:
            resize_img = sam_transform.apply_image(img)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
            input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
            batch_tensors.append(input_image)
        
        # Stack into single batch
        batch_input = torch.cat(batch_tensors, dim=0)
        
        # Encode batch
        with torch.no_grad():
            embeddings = sam_model.image_encoder(batch_input)
        
        # Store embeddings
        for j in range(embeddings.shape[0]):
            img_embeddings.append(embeddings[j].cpu().numpy())
    
    print(f'Processed {len(imgs)} images successfully')
    
    if len(imgs) > 0:
        imgs = np.stack(imgs, axis=0)
        gts = np.stack(gts, axis=0)
        img_embeddings = np.stack(img_embeddings, axis=0)
        boxes = np.array(boxes)
        tonguemasks = np.stack(tonguemasks, axis=0)

        save_file = join(save_path, f'{args.data_name}.npz')

        np.savez_compressed(save_file, imgs=imgs, boxes=boxes, gts=gts, img_embeddings=img_embeddings, tonguemasks=tonguemasks)
        print(f'Saved to {save_file}')
        print(f'  imgs: {imgs.shape}, gts: {gts.shape}, embeddings: {img_embeddings.shape}')
    else:
        print('No valid image pairs found!')


if __name__ == '__main__':
    # Load SAM model once

    # print(args.label_id) # arg.label_id = 1

    print(f"Loading SAM model ({args.model_type})...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    sam_model.eval()
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    print("Model loaded!")
    
    
    
    deal_fast(args.img_path, args.gt_path, sam_model, sam_transform)
    
    del sam_model
    print("Done!")
