import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import fitz
import json
import re
import math


def find_word_position(file_idx,pdf_path,json_path,png_dir,save_path):
    pdf = fitz.open(pdf_path)
    with open(json_path,'r') as fi:
        latex_json = json.load(fi)
    
    for page_idx in range(len(pdf)):
        # start a page
        page = pdf[page_idx]
        dct = {}

        # dct['image']
        png_path = f'{png_dir}_{page_idx}.png'    # data/train_data/lorempng/00_0.png
        with open(png_path, "wb") as f:
            f.write(pdf[page_idx].get_pixmap(dpi=96).pil_tobytes(format="PNG")) 
  
        dct['image'] = png_path.replace('data/train_data/','')
        
        # dct['pretext'],dct['prompt']
        fitz_color_dct = {}
        for b in page.get_textpage().extractDICT()['blocks']:
            for l in b['lines']:
                for s in l['spans']:
                    color = fitz.utils.sRGB_to_rgb(s['color'] )
                    color = tuple([math.ceil(c/5)*5 for c in color])
                    text = s['text'] 
                    x1,y1,x2,y2 = [round(n,2) for n in s['bbox']]
                    fitz_color_dct[color] = (text,[[x1,y1],[x2,y2]])
        pretext = []
        prompt = []
        try:
            if page_idx==0:
                if 'title' in latex_json.keys():
                    title = latex_json.pop('title')
                    lst = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',title)
                    for i in range(len(lst)):
                        c,t = (int(lst[i][0]),int(lst[i][1]),int(lst[i][2])),lst[i][3]
                        if i==0:
                            pretext.append(f'# {t}')
                        else:      
                            pretext.append(f' {t}')
                        # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                        position = fitz_color_dct[c][1]
                        prompt.append(position)
                    assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
                if 'author' in latex_json.keys():
                    author = latex_json.pop('author')
                    lst = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',author)
                    for i in range(len(lst)):
                        c,t = (int(lst[i][0]),int(lst[i][1]),int(lst[i][2])),lst[i][3]
                        if i==0:
                            pretext.append(f'\n\n{t}')  # 接title空行
                        else:      
                            pretext.append(f' {t}')
                        # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                        position = fitz_color_dct[c][1]
                        prompt.append(position)
                    assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
                if 'date' in latex_json.keys():
                    date = latex_json.pop('date')
                    date = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',date)[0]
                    c,t = (int(date[0]),int(date[1]),int(date[2])),date[3]
                    pretext.append(f'\n\n{t}') 
                    # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                    position = fitz_color_dct[c][1]
                    prompt.append(position)
                    assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
                if 'abstract' in latex_json.keys():
                    abstract = latex_json.pop('abstract')
                    lst = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',abstract)
                    for i in range(len(lst)):
                        c,t = (int(lst[i][0]),int(lst[i][1]),int(lst[i][2])),lst[i][3]
                        if i==0:
                            pretext.append(f'\n\n###### Abstract{t}')  
                        else:      
                            pretext.append(f' {t}')
                        # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                        position = fitz_color_dct[c][1]
                        prompt.append(position)
                    assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
                if 'keywords' in latex_json.keys():
                    keywords = latex_json.pop('keywords')
                    lst = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',keywords)
                    rect_keywords = pdf[page_idx].search_for("Keywords:")
                    if len(rect_keywords)>0:
                        x1,y1,x2,y2 = rect_keywords[0]
                        pretext.append('\n\nKeywords:')
                        prompt.append([[x1,y1],[x2,y2]])
                    for i in range(len(lst)):
                        c,t = (int(lst[i][0]),int(lst[i][1]),int(lst[i][2])),lst[i][3]
                        if i==0 and len(rect_keywords)==0:
                            pretext.append(f'\n\n{t}')  
                        else:      
                            pretext.append(f' {t}')
                        # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                        position = fitz_color_dct[c][1]
                        prompt.append(position)
                    assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
            if 'maintext' in latex_json.keys():
                maintext = latex_json['maintext']
                break_flag = False
                for para_idx,paragraph in enumerate(maintext):
                    if len(paragraph)>0:
                        lst = re.findall(r'\\color\[RGB\]{(\d+), (\d+), (\d+)}{(.*?)}',paragraph)
                        for word_idx in range(len(lst)):
                            c,t = (int(lst[word_idx][0]),int(lst[word_idx][1]),int(lst[word_idx][2])),lst[word_idx][3]
                            # assert t.strip()==fitz_color_dct[c][0].strip(), f"{t} didn't match the color!"
                            if c not in fitz_color_dct.keys():                            
                                break_flag = True
                                latex_json['maintext'] = latex_json['maintext'][para_idx:]  # 从当前段开始
                                index = latex_json['maintext'][0].find('\\color[RGB]{'+f'{c[0]}, {c[1]}, {c[2]}'+'}{'+t+'}')
                                latex_json['maintext'][0] = latex_json['maintext'][0][index:]# 当前段从当前word开始
                                break
                            position = fitz_color_dct[c][1]
                            prompt.append(position)
                            if i==0:
                                pretext.append(f'\n\n{t}')  
                            else:      
                                pretext.append(f' {t}')
                        if break_flag:
                            break
                        assert len(prompt)==len(pretext), f'len(prompt)={len(prompt)},while len(pretext)={len(pretext)}!'
            dct['prompt']=prompt
            dct['label']=[]
            dct['pretext']=pretext
    
            with open(save_path,'a')as fo:
                json.dump(dct,fo)
                fo.write('\n')
            print(f'{file_idx}:pdf:{pdf_path},page:{page_idx} done')
        except AssertionError as e:
            print(f'pdf:{pdf_path},page:{page_idx},{e}')
            continue

                        
            
        
           

if __name__ == "__main__":
    # python -m pdb nougat/dataset/find_position_by_color.py --color_file_dir data/lorempdf --png_dir data/train_data/lorempng --save_path data/train_data/train_lorem.jsonl
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_file_dir", type=str, required=True)
    parser.add_argument("--png_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    start_idx = 849
    
    for i,color_dir in enumerate(os.listdir(args.color_file_dir)[start_idx:]):
        json_path = os.path.join(args.color_file_dir,color_dir,'latex.json')
        pdf_path = os.path.join(args.color_file_dir,color_dir,'latex.pdf')
        png_path = os.path.join(args.png_dir,color_dir)
        find_word_position(i+start_idx,pdf_path,json_path,png_path,args.save_path) # data/train_data/lorempng/00
   

    coloriter = iter([(i,j,k) for i in range(20,220,5) for j in range(20,220,5) for k in range(20,220,5)])
    
        