import cv2
import numpy as np
import os
import argparse
import fitz
import orjson
import json

def find_blue_position(blue,origin):
    
    origin_png = 'data/to_find_position/origin.png'
    blue_png = 'data/to_find_position/blue.png'
    prompts=[]
    for i in range(len(origin)):
        with open(origin_png, "wb") as f:
            f.write(origin[i].get_pixmap().pil_tobytes(format="PNG"))
        with open(blue_png, "wb") as f:
            f.write(blue[i].get_pixmap().pil_tobytes(format="PNG"))
         
        origin_cv = cv2.imread(origin_png)
        origin_cv = cv2.resize(origin_cv, (672,896), interpolation= cv2.INTER_LINEAR)
        blue_cv = cv2.imread(blue_png)
        blue_cv = cv2.resize(blue_cv, (672,896), interpolation= cv2.INTER_LINEAR)
        if not (origin_cv[:,:,0]==blue_cv[:,:,0]).all():    # blue 不全相等时，判断是蓝色字体还是图片缺失
            arr1 = origin_cv[:,:,1]==blue_cv[:,:,1]         # green 相等的位置
            diff = np.where(origin_cv[:,:,0]!=blue_cv[:,:,0],arr1,~arr1)    # 取blue不相等、arr1相等的位置
            y,x = np.nonzero(diff)
            if len(x)>0:
                left_up = [min(x).item(),min(y).item()]     # convert np.int64 to int
                bottom_right = [max(x).item(),max(y).item()]
                prompt = [left_up,bottom_right]
                prompts.append((prompt,i))
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--find_position_path", type=str, help="PDF File", required=True)
    parser.add_argument("--train_path", type=str, help="train data path", required=True)
    parser.add_argument("--dpi", type=int, default=96)
    args = parser.parse_args()
    for blue_pdf in os.listdir(os.path.join(args.find_position_path,'blue')):
        stem = blue_pdf.split('_')[0]
        origin_pdf = os.path.join(args.find_position_path,'origin',stem+'.pdf')
        origin = fitz.open(origin_pdf)
        blue = fitz.open(os.path.join(args.find_position_path,'blue',blue_pdf))
        prompts = find_blue_position(blue,origin)
        for prompt,i in prompts:
            png_file = os.path.join(args.train_path,'png', f"{stem}_{i+1}.png" ) 
            with open(png_file, "wb") as f:
                f.write(origin[i].get_pixmap(dpi=args.dpi).pil_tobytes(format="PNG"))
            # pre_text只能手动找了
            with open(os.path.join(args.train_path,'train.jsonl'),'a') as f:
                json.dump({"image":png_file,"prompt":prompt,"markdown":''},f)
                f.write('\n')
        