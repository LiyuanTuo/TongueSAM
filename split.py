from pathlib import Path
from random import shuffle
import shutil
from PIL import Image, ImageOps
import os
train_path = Path("./data/train/train/")
val_path = Path("./data/train/valid/")

files = [f.name for f in Path("./compair/").iterdir() if f.is_file()]
shuffle(files)
print(len(files))  # check the number of files


train_files = files[:int(0.8*len(files))]
val_files = files[int(0.8*len(files)):]
# check the number of files in train and val sets
print(len(train_files), len(val_files))

os.system('pwsh -Command "rm ./imgraw/*"') 
os.system('pwsh -Command "Get-ChildItem -Path ./data/train -File -Recurse | Remove-Item -Force"') 

src_dir = Path("../舌苔抑郁症/img_rawdata/")
dest_dir = Path("./imgraw/")
for item in src_dir.rglob("*"):
    if item.is_file():
        shutil.copy2(item, dest_dir / item.name)

for f_name in os.listdir('./imgraw/'):
    if os.path.splitext(f_name)[1].lower() in ['.jpg', '.jpeg', '.webp']:
        f_path = os.path.join('./imgraw/', f_name)
        img = Image.open(f_path)
        img = ImageOps.exif_transpose(img)

        os.remove(f_path) 

        img = img.convert('RGB')
        new_path = os.path.splitext(f_path)[0] + '.png'
        img.save(new_path)
        


def move_files(file_list, target_base):
    for f_name in file_list:

        # ground truth
        src_gt = Path("../舌苔抑郁症/coating_mask") / f_name
        shutil.copy(str(src_gt), str(target_base / "gt" / f_name))

        # raw image  imgraw 文件夹中 的图片的名字 = coating_mask = mask = compair 文件夹中的图片的名字
        src_img = Path("./imgraw") / f_name
        shutil.copy(str(src_img), str(target_base / "img" / f_name))


move_files(train_files, train_path)

move_files(val_files, val_path)
