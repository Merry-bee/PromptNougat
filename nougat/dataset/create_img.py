# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import random
import os
from pathlib import Path
import re
import json

# 生成一张空白图片
def create_bk_img(save_path):
    width = 672
    height = 896
    img = np.full([width, height, 3], 255,dtype=np.uint8)
    img = cv2.resize(img, (672, 896), interpolation=cv2.INTER_CUBIC)
    # 展示图片
    # cv2.namedWindow('image')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(save_path, img)

# 在图片中写入文字
def write_pic(bk_img_path,text_path,save_dir):
    # 设置需要显示的字体
    bk_img = cv2.imread(bk_img_path)
    with open(text_path,'r') as fi:
        texts = fi.read().replace('\n','')[2000:]
    
    # formula1 = re.findall(r'\\\[.*?\\\]',texts) # 行间公式
    texts = re.sub(r'\\\[.*?\\\]','',texts)
    # formula2 = re.findall(r'\\\(.*?\\\)',texts) # 行内公式
    texts  = re.sub(r'\\\(.*?\\\)','',texts)
    texts  = texts.split()  # 以word为单位

    word_idx = 0
    page_idx = 0
    while word_idx<len(texts)-100:
        # start to create a png
        pretext = ''
        chessboard = []
        num_rows = random.randint(2,4) # 每页纵向分为2-4块
        for row in range(num_rows):
            num_cols = random.randint(1,3)  # 每块随机分为1-3栏
            for col in range(num_cols):
                # 记录每块的左上角位置、宽度和高度
                chessboard.append({'x':col*bk_img.shape[1]//num_cols,'y':row*bk_img.shape[0]//num_rows,
                                'num_cols':num_cols,'num_rows':num_rows})
        # 绘制文字信息
        font1 = ImageFont.truetype('data/train_data/create_png/TIMES.TTF', 15)  # 字体搜索路径：Windows/Fonts
        font2 = ImageFont.truetype('data/train_data/create_png/TIMES.TTF', 13)
        img_pil = Image.fromarray(bk_img)
        draw = ImageDraw.Draw(img_pil)
        save_path = os.path.join(save_dir,f'{Path(text_path).name}_{page_idx}.png')
        for _ in range(len(chessboard)):
            # start to write a board
            board = chessboard.pop(random.randint(0,len(chessboard)-1))   # 随机选取一块
            text_str = ''
            num_words_a_row = 12//board['num_cols']
            num_rows_a_board = 50//board['num_rows']

            # a data per board
            dct = {"image": save_path.replace('data/train_data/',''), "prompt": [[board['x']+10,board['y']+20], [board['x']+bk_img.shape[1]//board['num_cols'],board['y']+20+15]], "label": ' '.join(texts[word_idx:word_idx+num_words_a_row]), "pretext": pretext}
            with open(os.path.join(save_dir,'train_position.jsonl').replace('create_png/',''), "a") as f:
                json.dump(dct,f)
                f.write('\n')

            # 当前board要写的字
            text_lst = texts[word_idx:word_idx+num_words_a_row*num_rows_a_board]
            for row in range(1,num_rows_a_board+1):
                text_str += ' '.join(text_lst[num_words_a_row*(row-1):num_words_a_row*row])+'\n'
            draw.text((board['x']+10,board['y']+20), text_str, font=font2, fill=(0, 0, 0))
            # 下一个board开头的word
            word_idx = word_idx+num_words_a_row*num_rows_a_board  
            if word_idx > len(texts)-5:
                break  
            pretext = ' '.join(texts[:word_idx])
            
        new_img = np.array(img_pil)
        
        
        cv2.imwrite(save_path, new_img)
        page_idx += 1


if __name__ == '__main__':
    bk_img_path = 'data/train_data/create_png/bk_img.png'
    # create_bk_img(save_path = bk_img_path)
    save_dir = 'data/train_data/create_png'
    text_dir = 'output/greedy_search/correct'
    for i,text_file in enumerate(os.listdir(text_dir)[26:]):
        if i>50:
            break
        write_pic(bk_img_path,text_path=os.path.join(text_dir,text_file),save_dir=save_dir)