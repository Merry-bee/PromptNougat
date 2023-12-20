import re
import json
import random
import datetime
import os
from pathlib import Path
import fitz
from PIL import Image
import math
from collections import OrderedDict


with open('nougat/p_dataset/Greek.json','r') as fi:
    GREEK = json.loads(fi.read())
with open('nougat/p_dataset/MathSymbol.json','r') as fi:
    MATH = json.loads(fi.read())
        
    
def random_date():
    year = random.randint(1800,2025)
    month = random.randint(1,12)
    day = random.randint(1,31)
    random_date = datetime.date(year,month,day).strftime('%B %d, %Y')
    return random_date





def lined_file(tex_file):
    
    with open(tex_file,'r',encoding='utf-8') as fi:
        lines = fi.readlines()
    # 删除注释和不必要的定义头
    lines = [line for line in lines if not line.strip().startswith('%')]
    
    # split lines to new_lines
    new_lines = []
    begin_flag = False  # 忽略掉tex开头的预定义部分
    for i,line in enumerate(lines):
        if begin_flag and not any(word in line for word in ['\\def','\\vskip','\\vspace','\\voffset']):
            # TODO: 公式内部可以再细分; 还有一些不能加的行，如&；mmd被标黄的部分好像没有翻译为markdown而是字符本身
            new_lines.extend(line.strip().split(' '))
        else:
            new_lines.append(line.strip())
            if ('\\begin{document}') in line:
                begin_flag = True

    # 添加行号
    for i,line in enumerate(new_lines):
        if line.strip() and not line.strip().startswith('\\') and not line.strip().startswith('{') and not line.strip().endswith('}') and not line.strip().endswith('\\'):
            new_lines[i] = line.replace(line.strip(),line.strip()+f',,{"{:06}".format(i)}')
    
    with open(tex_file.replace('arxiv_origin','arxiv_line'),'w',encoding='utf-8') as fo:
        fo.writelines([line+'\n' for line in new_lines])
               
        
def lined_dct(mmd_file):
    # dct: {line:word}
    dct_line = OrderedDict()
    with open(mmd_file,'r') as fi:
        mmd = fi.read()
    mmds = mmd.split(' ')#[mmd for mmd in re.split(r'(?=\S+,,\d{6})|\s',mmd) if mmd]
    line = 0
    for mmd in mmds:
        res = re.match(r'(\S+),,(\d{6})',mmd)
        if res:
            word = res.group(1)
            line = int(res.group(2))
            dct_line[line] = word
        
        else:   
            # not included：对上一个line进行递增，目的是避免键的重复
            line += 0.01
            dct_line[line] = mmd
    gt = ' '.join(dct_line.values())
    with open(mmd_file.replace('line.mmd','clean.mmd'),'w') as fo:
        fo.write(gt)
    return dct_line




def match_datapair(mmd_line_dct,pdf_file,png_dir,save_data_path,page_rect=(0,0,672,896)):
    '''
    TODO:
    转为黑白图片
    '''
    pdf = fitz.open(pdf_file)
    
    for page_idx in range(len(pdf)):
        # start a page
        page = pdf[page_idx]
        page.set_mediabox(fitz.Rect(page_rect))
        
        png_path = f'{png_dir}_{page_idx}.png'    # data/arxiv_train_data/png/0710.2897/0710.2897_0.png
        os.makedirs(Path(png_path).parent,exist_ok=True)
        with open(png_path, "wb") as f:
            f.write(page.get_pixmap(dpi=300).pil_tobytes(format="PNG")) 
        dct = {}
        # dct['image']
        dct['image'] = png_path.replace('data/arxiv_train_data/','')
        
        # dct['pretext'],dct['prompt']
        # fitz_color_dct: {(r,g,b):(t,p)}
        fitz_color_dct = {}
        ref_lst = []    # 参考文献
        affiliation_lst = []    # affiliation
        for b in page.get_textpage().extractDICT()['blocks']:
            for l in b['lines']:
                for s in l['spans']:
                    color = fitz.utils.sRGB_to_rgb(s['color'] )
                    color = tuple([math.ceil(c/5)*5 for c in color])
                    text = s['text'] 
                    if not re.search(r'\S',text):   # 无内容的忽略，如' '
                        continue
                    x1,y1,x2,y2 = [round(n,2) for n in s['bbox']]
                    if ref_lst:
                        ref_lst.append((text,[[x1,y1],[x2,y2]]))
                    else:
                        # affiliation
                        if page_idx ==0 and color == (0,0,0) and len(text)>10 and y1<page.rect[3]//2:
                            for word in text.split(' '):
                                word_rects = page.search_for(word)
                                if len(word_rects) == 1:
                                    affiliation_lst.append((word,[[word_rects[0][0],word_rects[0][1]],[word_rects[0][2],word_rects[0][3]]]))
                                else:
                                    affiliation_lst.append((word,["mask"]))
                        # Reference
                        elif page_idx >= len(pdf)-2 and color == (0,0,0) and text == 'References':
                            ref_lst.append((text,[[x1,y1],[x2,y2]]))
                        # 发生换行时会有两个token对应同一个color，此时只保留第一个position
                        elif color not in fitz_color_dct.keys():
                            fitz_color_dct[color] = (text,[[x1,y1],[x2,y2]])
                                
        pretext = []
        prompt = []
        
        # fitz_color_dct: 当页内容(color升序<=>latex顺序)，mmd_line_dct：全文内容(mmd顺序)
        fitz_color_dct = OrderedDict(sorted(fitz_color_dct.items(),key=lambda x:(x[0][0],x[0][1],x[0][2])))
        if (0,0,0) in fitz_color_dct.keys():
            fitz_color_dct.pop((0,0,0))
       
      
        while len(fitz_color_dct) > 0:
            
            # ground_truth: mmd
            mmd_color = next(iter(mmd_line_dct))     
            mmd_text = mmd_line_dct[mmd_color]
           

            if mmd_color in fitz_color_dct.keys():
             
                pretext.append(' '+mmd_text)
                prompt.append(fitz_color_dct[mmd_color][1])
                mmd_line_dct.popitem(last=False)
                fitz_color_dct.pop(mmd_color)
            # affiliation不在fitz_color_dct中，但在mmd_line_dct中
            elif affiliation_lst and mmd_text == affiliation_lst[0][0] and inserted_color(mmd_color):
                pretext.append(' '+mmd_text)
                prompt.append(affiliation_lst[0][1])
                affiliation_lst = affiliation_lst[1:]
                mmd_line_dct.popitem(last=False)
            else:
                # 非显式符号:需要确保当前段落确实在该页面（图表重排）
                # TODO: 这里Position先作为mask, 实际最好输入下一个词的位置，因为此类大多数为角标引用等
                
                # 找到下一个显式符号
                for c in iter(mmd_line_dct):
                    if not inserted_color(c):
                        if c in fitz_color_dct.keys():
                            # 还是当前页面：先添加mmd,fitz不动
                            pretext.append(' '+mmd_text)
                            prompt.append(['mask'])
                            mmd_line_dct.popitem(last=False)
                        else:
                            # mmd到下一页了，按照fitz遍历

                            # 非第一页的开头，存在不在当前页面的mmd_color，说明是上一页遗留的无效字符
                            if page_idx and len(pretext)<100:
                                mmd_line_dct.pop(mmd_color)
                            fitz_color = next(iter(fitz_color_dct))
                            if fitz_color in mmd_line_dct.keys():
                                pretext.append(' '+mmd_line_dct[fitz_color])
                                mmd_line_dct.pop(fitz_color)
                            else:   # 一般不会走到这里
                                pretext.append(' '+fitz_color_dct[fitz_color][0])
                            prompt.append(fitz_color_dct[fitz_color][1])
                            fitz_color_dct.pop(fitz_color)
                        break
        
        # 此时len(fitz_color_dct)=0，已跳出循环
        for text,position in ref_lst:    
            pretext.append(text)
            prompt.append(position)
                    
        dct["prompt"] = prompt
        dct["label"] = []
        dct["pretext"] = pretext
        with open(save_data_path,'a') as fo:
            json.dump(dct,fo)
            fo.write('\n')
            
           
  
            
            
         
            

def main(fold_lst,start_idx):
    
    for i,tex_fold in enumerate(fold_lst[start_idx:]):
        break_flag=False
        try:
            os.makedirs(f'data/arxiv_all_files/{tex_fold}', exist_ok=True)
            file_idx = i+start_idx
            # 复制tex文件夹: 保留两份
            if os.system(f'cp -r -n ~/../mnt/data/oss_beijing/zhonghansen/arxiv/latex/{tex_fold} data/arxiv_origin/') \
                or os.system(f'cp -r data/arxiv_origin/{tex_fold} data/arxiv_line/'):
                print(f'{file_idx}. Copy error:{tex_fold}')
                continue
            # 找到所有.tex文件
            tex_files = os.popen(f'find data/arxiv_origin/{tex_fold} -type f -name "*.tex"').read().split('\n')[:-1]
            if not tex_files:
                tex_files = os.popen(f'find data/arxiv_origin/{tex_fold}/{tex_fold} -maxdepth 1 -type f').read().split('\n')[:-1]
                tex_files = [str(Path(tex_file).rename(tex_file+'.tex')) for tex_file in tex_files]
                if not tex_files:
                    print(f'{file_idx}. tex not exist: {tex_fold}')
                    break_flag = True
                    continue 
            # \textcolor
            for tex_file in tex_files:
                try:
                    os.system(f'cp {tex_file} data/arxiv_all_files/{tex_fold}/origin.tex')
                    lined_file(tex_file)
                    os.system(f'cp {tex_file.replace("arxiv_origin","arxiv_line")} data/arxiv_all_files/{tex_fold}/{Path(tex_file).name}')
                except ValueError as e:
                    print(f'{file_idx}. lined_file error: {tex_fold} :{e}')
                    break_flag = True
                    break 
            if not break_flag:
                # 找到主文件
                tex_file = os.popen(f'find data/arxiv_line/{tex_fold} -maxdepth 1 -type f -name "*.tex"').read().split('\n')[0]
                # latexmk: tex->pdf
                os.system(f'latexmk -pdfps -f -g -interaction=nonstopmode -cd data/arxiv_line/{tex_fold} -output-directory=outputs {tex_file} > log/construct_data.txt 2>&1')
                pdf_file = f'data/arxiv_line/{tex_fold}/outputs/{str(Path(tex_file).with_suffix(".pdf").name)}'
                if not os.path.exists(pdf_file):
                    pdf_file = os.popen(f'find data/arxiv_line/{tex_fold}/outputs -maxdepth 1 -type f -name "*.pdf" ').read().split('\n')[0]
                if not os.path.exists(pdf_file):
                    print(f'{file_idx}. tex 2 pdf fail: {tex_fold}')
                    break_flag = True
                # else:
                #     os.system(f'cp {pdf_file} data/arxiv_all_files/{tex_fold}/{Path(pdf_file).name}')

                    
            if not break_flag:           
                # tex->html
                os.system(f'latexmlc --nocomments --includestyles --dest=data/arxiv_line/{tex_fold}/outputs/color.html --path=data/arxiv_line/{tex_fold} {str(Path(tex_file).name)} > log/construct_data.txt 2>&1')
                html_file = f'data/arxiv_line/{tex_fold}/outputs/color.html'
                if not os.path.exists(html_file):
                    print(f'{file_idx}. tex 2 html fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {html_file} data/arxiv_all_files/{tex_fold}/{Path(html_file).name}')
            if not break_flag:    
                # html->mmd
                os.system(f'python nougat/dataset/parser/html2md.py --html data/arxiv_line/{tex_fold}/outputs/color.html --out data/arxiv_line/{tex_fold}/outputs/color.mmd > log/construct_data.txt 2>&1')
                mmd_file = f'data/arxiv_line/{tex_fold}/outputs/color.mmd'
                if not os.path.exists(mmd_file):
                    print(f'{file_idx}. html 2 mmd fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {mmd_file} data/arxiv_all_files/{tex_fold}/{Path(mmd_file).name}')
                
            if not break_flag:
                # mmd->dct_line
                mmd_file = f'data/arxiv_line/{tex_fold}/outputs/color.mmd'
                dct_line = lined_dct(mmd_file)
                os.system(f'cp {mmd_file.replace("color.mmd","clean.mmd")} data/arxiv_all_files/{tex_fold}/')
                
                # construct datapair
                match_datapair(dct_line,pdf_file,png_dir=f'data/arxiv_train_data/png/{tex_fold}/{tex_fold}',save_data_path='data/arxiv_train_data/train.jsonl')
                print(f'{file_idx}. html 2 mmd fail: {tex_fold}')
            # 删除tex文件夹
            os.system(f'rm -r data/arxiv_origin/{tex_fold}') 
            os.system(f'rm -r data/arxiv_line/{tex_fold}')
        except Exception as e:
            print(f'{file_idx}. {tex_fold}:{e}')
            continue
            

    
if __name__ == '__main__':
    # lined_dct('data/arxiv_line/0704.0001/outputs/color.mmd')
    # lined_file('data/arxiv_origin/0704.0001/origin.tex')
    with open('data/pdf_list/latex_dir.txt','r') as fi:
        lines = fi.readlines()
    fold_lst = [line.strip() for line in lines]
    main(fold_lst,start_idx=0)
    