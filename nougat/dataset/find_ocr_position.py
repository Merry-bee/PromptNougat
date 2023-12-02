import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import fitz
import orjson
import json
import re
from bs4 import BeautifulSoup
from markdown import markdown
import Levenshtein
import random

def save_jsonl(page,jsonl_path,png_path,prompt,next_token,pre_texts):
    
    with open(png_path, "wb") as f:
        f.write(page.get_pixmap(dpi=96).pil_tobytes(format="PNG"))
    relative_png_path = str(Path(png_path).relative_to('data/train_data'))
    with open(jsonl_path, "a") as f:
        json.dump({"image":relative_png_path, "prompt": prompt, "label": next_token, "pretext":pre_texts},f)
        f.write('\n')

def find_ocr_position(json_path,pdf_path,jsonl_path,save,image_size=[896, 672]):
    with open(json_path,'r') as fi:
        lines = fi.readlines()
    pdf =  fitz.open(pdf_path)
    for line in lines:
        if random.random()>0.2: # 以1/5的概率抽样
            continue
        dct = orjson.loads(line)
        pre_texts = dct["pre_texts"]
        next_token = dct["predicted_token"]
        page = pdf[dct["page"]-1]

        w=180
        next_rects = page.search_for(next_token)
        result = None
        # if next_rects能找到，才能做label，page.get_text(clip=前面的区域)
        min_ratio = 0.3
        pattern = re.compile(r'\^|_|{|}|(\\sim)|(\\alpha)|(\\omega)|(\\rm)|(\\)',re.I)
        if len(next_rects)>0:
            # print(f'next_token:{next_token}')
            pre_html = markdown(pre_texts)
            pre_raw = ''.join(BeautifulSoup(pre_html).findAll(text=True))
            pre_raw = re.sub(pattern,'',pre_raw)
            for next_rect in next_rects:    # next_rect的位置
                x1,y1,x2,y2=next_rect
                pre_clip_left = fitz.Rect(x1-w,y1,x1,y2)
                pre_text_left = page.get_text(clip=pre_clip_left)
                pre_text_left = pre_text_left.strip().split('\n')[-1]
                if len(pre_text_left)>0:
                    length = min(len(pre_text_left),len(pre_raw))
                    distance = Levenshtein.distance(pre_raw[-length:],pre_text_left[-length:])
                    if length>0 and len(pre_text_left)>4 and distance/length < min_ratio:
                        min_ratio = distance/length
                        result = next_rect
                else:   # next_token左侧为空，可能在换行位置，匹配上一行
                    pre_clip_up = fitz.Rect(x1,y1-(y2-y1),page.rect[2],y1)
                    pre_text_up = page.get_text(clip=pre_clip_up)
                    for pre_text_up in pre_text_up.strip().split('\n'):   # 上方一整行，可能分几栏
                        length = min(len(pre_text_up),len(pre_raw))
                        distance = Levenshtein.distance(pre_raw[-length:],pre_text_up[-length:])
                        if length>0 and len(pre_text_up)>4 and distance/length < min_ratio:
                            min_ratio = distance/length
                            result = next_rect
        
       
        png_path = os.path.join(args.png_dir, f'{Path(json_path).stem}_{dct["page"]}.png')
        '''if result is not None and save:
            x1,y1,x2,y2=result
            # resize to image_size
            prompt = [[x1/page.rect.width*image_size[1],y1/page.rect.height*image_size[0]],[x2/page.rect.width*image_size[1],y2/page.rect.height*image_size[0]]]
            save_jsonl(page,jsonl_path,png_path,prompt,next_token,pre_texts)  '''
        if result is None and save:
            prompt = []
            save_jsonl(page,jsonl_path,png_path,prompt,next_token,pre_texts)  
        





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--png_dir", type=str, required=True)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    # python -m pdb nougat/dataset/find_ocr_position.py --json_dir data/quantum/json --jsonl_path data/train_data/train1.jsonl --png_dir data/train_data/png1 --save True
    if os.path.isdir(args.json_dir):
        for json_file in os.listdir(args.json_dir):
            json_path = os.path.join(args.json_dir,json_file)
            pdf_path = os.popen(f'find {Path(args.json_dir).parent} -name {Path(json_file).with_suffix(".pdf")}').read().strip()
            find_ocr_position(json_path,pdf_path,args.jsonl_path,save=args.save)
    # python -m pdb nougat/dataset/find_ocr_position.py --json_dir data/to_find_position/json/0704.0118.json --jsonl_path data/train_data/train.jsonl --png_dir data/train_data/png --save False
    elif os.path.isfile(args.json_dir):
        json_path = args.json_dir
        pdf_dir = Path(args.json_dir).parents[1] / 'origin'
        pdf_path = os.path.join(pdf_dir,Path(json_path).stem+'.pdf')
        result = find_ocr_position(json_path,pdf_path,args.jsonl_path,save=args.save)

    

        
    
    

            