import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm_notebook

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def verify_path(path):
    isFile = os.path.isfile(path)
    isDir = os.path.isdir(path)
    return isFile | isDir

def get_images(root_path, sub_path):
    full_path = root_path + sub_path
    assert verify_path(full_path), "File is not exist"
    
    result = []
    
    if os.path.isdir(full_path):
        sub_files = os.listdir(full_path)
        for path in sub_files:
            result += get_images(root_path, sub_path+"/"+path)
    else:
        result.append(sub_path)
    return result

def resize_image(in_path, out_path, scale=2):
    assert os.path.isfile(in_path)
    assert not os.path.isfile(out_path)
    
    img = Image.open(in_path)
    img_resize = img.resize((img.width // scale, img.height // scale), Image.LANCZOS)
    img_resize.save(out_path)
    img.close()

    
if __name__ =='__main__':
    
    scale = 2
    root_path = "./train"
    image_dir = "/images"
    out_path = "./rescale_x{}".format(scale)

    # !!!! 주의해야함 폴더가 삭제됨 !!!!
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    # !!!! 주의해야함 폴더가 삭제됨 !!!!

    images = get_images(root_path, image_dir)

    for img_sub_path in tqdm_notebook(images):
        image_dir, image_filename = os.path.split(img_sub_path)
        createFolder(out_path+image_dir)
        resize_image(root_path+img_sub_path, out_path+img_sub_path, scale)