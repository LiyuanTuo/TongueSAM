import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 Tkinter 错误
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
from monai.losses import DiceCELoss
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
from skimage import io
from  utils_metrics import *
from skimage import transform, io, segmentation
from segment.yolox import YOLOX
import random
import math
from functools import partial
##############################################################################################################
num_epochs = 15
ts_npz_path='./data/test_npz/'
npz_tr_path = './data/train_npz/'
model_type = 'vit_b'
checkpoint = './pretrained_model/sam.pth'
device = 'cuda:0'
model_save_path = './logs/'
if_save=True
if_onlytest=False
batch_size=32
prompt_type='mask'
lr_decay_type= "cos"
Init_lr= 1e-4
point_num=3
segment=None
###############################################################################################################    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):            
        self.npz_data=np.load(data_root)
        self.ori_gts = self.npz_data['gts']  # 这里已经能保证gt 中 1 是 舌苔 0 是其他
        self.img_embeddings = self.npz_data['img_embeddings']
        self.imgs=self.npz_data['imgs']
        self.model=segment        
        self.point_num=point_num
        self.tonguemask = self.npz_data['tonguemasks']
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):           
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        img=self.imgs[index]
        H, W = gt2D.shape         
    
           
        tongue_mask_256 = cv2.resize(self.tonguemask[index].astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(tongue_mask_256[None,:,:]).float()
#####################################################Begin############################################################################
Min_lr=Init_lr*0.01
lr_limit_max    = Init_lr 
lr_limit_min    = 3e-4 
Init_lr_fit     = min(max(batch_size / batch_size * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit      = min(max(batch_size / batch_size * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, num_epochs)
train_losses = []
val_losses = []
best_iou=0
best_pa=0
best_acc=0


sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')  # %% train
os.makedirs(model_save_path, exist_ok=True)
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=Init_lr_fit, weight_decay=0)

for epoch in range(num_epochs):  
    print(f'EPOCH: {epoch}')   
    epoch_loss = 0    
###############################################################Test##################################################################
    sam_model.eval()
    val_gts=[]
    val_preds=[]    
    with torch.no_grad():                                                        
        for f in os.listdir(ts_npz_path):                             
            ts_dataset = NpzDataset(join(ts_npz_path,f))            
            ts_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=True)
            for step, (image_embedding, gt2D, tonguemask) in enumerate(ts_dataloader):                                                                                                                           
                if prompt_type=='mask':  
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=tonguemask.to(device),
                    )
                # 以下代码必须在 for step 循环内！
                mask_predictions, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                    image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                    multimask_output=False,
                )                                                                                          
                for i in range(mask_predictions.shape[0]):
                    mask = mask_predictions[i]
                    mask = mask.cpu().detach().numpy().squeeze()
                    # 调试：打印第一个epoch第一个batch的logit范围和GT分布
                    if epoch == 0 and step == 0 and i == 0:
                        print(f"[DEBUG] logit range: [{mask.min():.4f}, {mask.max():.4f}]")
                        print(f"[DEBUG] gt unique: {np.unique(gt2D[i].cpu().numpy())}")
                    mask = cv2.resize((mask > 0).astype(np.uint8),(gt2D.shape[3], gt2D.shape[2]))  # logit > 0 等价于 sigmoid(logit) > 0.5                                                      
                    gt_data=gt2D[i].cpu().numpy().astype(np.uint8)                 
                    val_gts.append(gt_data.astype(np.uint8))
                    val_preds.append(mask.astype(np.uint8))                          
            # 解决内存泄漏：每遍历完一个npz文件，手动关闭并强制回收内存
            ts_dataset.npz_data.close()
            del ts_dataset, ts_dataloader
            import gc
            gc.collect()
        iou,pa,acc=compute_mIoU(val_gts,val_preds) 
        if  iou> best_iou:
            best_iou=iou            
            best_pa=pa
            best_acc=acc        
            if if_onlytest:
                continue
            if if_save==True:
                torch.save(sam_model.state_dict(), join(model_save_path, 'best.pth'))# plot loss                  
        print('best_miou:'+str(best_iou))
        print('best_pa:'+str(best_pa))
        print('best_acc:'+str(best_acc))
        if if_onlytest:
                continue
###############################################################Train##################################################################
    sam_model.train()
    lr = lr_scheduler_func(epoch)    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for f in os.listdir(npz_tr_path):                                     
        train_dataset = NpzDataset(join(npz_tr_path,f))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for step, (image_embedding, gt2D, tonguemask) in enumerate(train_dataloader):                                                
            with torch.no_grad():
                if prompt_type=='mask':  
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=tonguemask.to(device),
                    )
            # 以下代码必须在 for step 循环内！
            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_predictions = F.interpolate(mask_predictions, size=(gt2D.shape[2],gt2D.shape[3]), mode='bilinear', align_corners=False)
            gt2D = gt2D.to(device).float()
            loss = seg_loss(mask_predictions, gt2D)
            # 调试：打印每个epoch的第一个batch的loss和梯度信息
            if step == 0:
                print(f"  [TRAIN] epoch={epoch} lr={lr:.6f} loss={loss.item():.6f}")
                print(f"  [TRAIN] pred range: [{mask_predictions.min().item():.4f}, {mask_predictions.max().item():.4f}]")
                print(f"  [TRAIN] gt sum: {gt2D.sum().item():.0f} / {gt2D.numel()} ({gt2D.sum().item()/gt2D.numel()*100:.1f}%)")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sam_model.mask_decoder.parameters(), max_norm=1.0)  # 梯度裁剪，防止权重爆炸
            # 调试：检查梯度是否存在
            if step == 0:
                total_grad = 0
                for p in sam_model.mask_decoder.parameters():
                    if p.grad is not None:
                        total_grad += p.grad.abs().mean().item()
                print(f"  [TRAIN] avg grad magnitude: {total_grad:.8f}")
            optimizer.step()
            epoch_loss += loss.item()
        
        # 解决内存泄漏：手动关闭释放底层大数组，避免撑爆内存 (OOM)
        train_dataset.npz_data.close()
        del train_dataset, train_dataloader
        import gc
        gc.collect()

    train_losses.append(epoch_loss)    
################################################################################################################################  
if if_onlytest is False:
    plt.figure()
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('train_loss')
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()
    print(f'Training complete! Best mIoU: {best_iou:.2f}%')