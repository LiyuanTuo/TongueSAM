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

# set up the parser
parser = argparse.ArgumentParser(description='preprocess grey and RGB images (fast version)')
parser.add_argument('-i', '--img_path', type=str, required=True, help='path to the images')
parser.add_argument('-gt', '--gt_path', type=str, required=True, help='path to the ground truth (gt)')
parser.add_argument('-o', '--npz_path', type=str, required=True, help='path to save the npz files')
parser.add_argument('--data_name', type=str, default='tongue', help='dataset name')
parser.add_argument('--image_size', type=int, default=400, help='image size')
parser.add_argument('--img_name_suffix', type=str, default='.jpg', help='image name suffix')
parser.add_argument('--label_id', type=int, default=1, help='label id')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='./pretrained_model/sam.pth', help='checkpoint')
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


def process_single_image(img_path, gt_path, image_name, gt_name, image_size, label_id, no_augment=True):
    """Process a single image and return preprocessed data"""
    gt_data = io.imread(join(gt_path, gt_name))
    # DEBUG: Print unique values to verify label_id
    # print(f"Processing {image_name}, GT unique: {np.unique(gt_data)}, Target label: {label_id}")

    image_data = io.imread(join(img_path, image_name))
    
    # Convert bool mask to uint8
    if gt_data.dtype == bool:
        gt_data = gt_data.astype(np.uint8)
    
    if no_augment:
        image_data, gt_data = simple_preprocess(image_data, gt_data, image_size)
    else:
        # Augmentation (rotation-focused)
        h, w = image_data.shape[:2]
        # crop_size = min(h, w, 300)
        # top = np.random.randint(0, max(1, h - crop_size))
        # left = np.random.randint(0, max(1, w - crop_size))
        # image_data = image_data[top:top+crop_size, left:left+crop_size]
        # gt_data = gt_data[top:top+crop_size, left:left+crop_size]

        # Random rotation (same angle for image and mask)
        angle = np.random.uniform(-135, 135)
        image_data, gt_data = rotate_image_and_mask(image_data, gt_data, angle)

        image_data = cv2.resize(image_data, (image_size, image_size))
        gt_data = cv2.resize(gt_data, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    
    # Ensure gt is 2D
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    
    # Binarize gt 
    gt_data = (gt_data > 0).astype(np.uint8) # fixed this serious bug
    # print(f"Post-Binarization unique: {np.unique(gt_data)}")
    
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
    
    # Get bounding box from gt
    y_indices, x_indices = np.where(gt_data > 0)
    if len(x_indices) > 0:
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        box = np.array([xmin, ymin, xmax, ymax])
    else:
        box = np.array([0, 0, image_size-1, image_size-1])
    
    return image_data, gt_data, box


def deal_fast(img_path, gt_path, sam_model, sam_transform):
    """Optimized processing with batch encoding"""
    names = sorted(os.listdir(gt_path))
    save_path = args.npz_path
    os.makedirs(save_path, exist_ok=True)
    print(f'Processing {len(names)} images...')
    
    imgs = []
    gts = []
    boxes = []
    img_embeddings = []
    
    # Collect all preprocessed images first
    batch_images = []
    batch_indices = []
    
    for idx, gt_name in enumerate(tqdm(names, desc="Loading images")):
        image_name = gt_name.split('.')[0] + args.img_name_suffix
        try:
            image_data, gt_data, box = process_single_image(
                img_path, gt_path, image_name, gt_name, 
                args.image_size, args.label_id, args.no_augment
            )
            imgs.append(image_data)
            gts.append(gt_data)
            boxes.append(box)
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
        
        save_file = join(save_path, f'{args.data_name}.npz')

        np.savez_compressed(save_file, imgs=imgs, boxes=boxes, gts=gts, img_embeddings=img_embeddings)
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
