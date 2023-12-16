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
        
COLORS = [(r,g,b) for r in range(5,255,5) for g in range(0,255,5) for b in range(0,255,5)]
color_idx = 0

def greater_rgb(color1,color2):
    '''
    color1:(r,g,b)
    color2:(r,g,b)
    '''
    if color1[0]==color2[0]:
        if color1[1] == color2[1]:
            return color1[2]>color2[2]
        else:
            return color1[1]>color2[1]
    else:
        return color1[0]>color2[0] 
    
def add_rgb(color,increment=0.1):
    # 对(R,G,B)计算加法，递增序列
    if color[2]+increment <= 255:
        return (color[0],color[1],color[2]+increment)
    elif color[1]+increment <= 255:
        return (color[0],color[1]+increment,0)
    else:
        return (color[0]+increment,color[1],color[2])
def inserted_color(color):
    # 判断color是否为后加入的，即不可见符号
    for i in range(3):
        if color[i]%5:
            return True
    return False
    
def random_date():
    year = random.randint(1800,2025)
    month = random.randint(1,12)
    day = random.randint(1,31)
    random_date = datetime.date(year,month,day).strftime('%B %d, %Y')
    return random_date

def colored_word(word):
    global color_idx
    color = f"{'{:03}'.format(COLORS[color_idx][0])},{'{:03}'.format(COLORS[color_idx][1])},{'{:03}'.format(COLORS[color_idx][2])}"
    color_idx  = (color_idx+1)%len(COLORS)
    color_word = f'\\textcolor[RGB]{{{color}}}{{{word}}}'
    return color_word

def colored_text(text):    
    
    texts = [seg for seg in re.split(r'(\\[A-Za-z]+(?:\[.*?\])*(?:\{.*?\})+)|\s', text) if seg]  # 使用\tag[]{}或\tag{}或空白符将句子分为单词，括号内不会被切分
    # seg: 单词、公式等切分片段
    for j,seg in enumerate(texts):
        if not seg:
            continue
        seg_idx = 0
        new_seg = ''
        while seg_idx < len(seg):            
            # \tag
            tag_res = re.match(r'\\[A-Za-z]+',seg[seg_idx:])
            if tag_res:
                word = tag_res.group(0)
                # 加颜色显示的tag: \alpha \sum
                if word in GREEK.keys() or word in MATH:
                    new_seg += colored_word(word)
                    seg_idx += tag_res.span()[1]
                # 需要和后面括号内容一起保留原样的tag
                elif word in ['\\label','\\begin','\\end','\\includegraphics','\\resizebox','\\cline','\\multicolumn','\\multirow','\\affiliation','\\pagestyle','\\email',
                              '\\input','\\bibliographystyle','\\bibliography','\\newcommand','\\usepackage','\\preprint','\\cite','\\citep','\\ref','\\url','\\bibitem','\\bibinfo','\\bibnamefont']:
                    ignore_res = re.match(r'\\[A-Za-z]+(?:\[.*?\])*(?:\{.*?\}){1,2}',seg[seg_idx:])
                    new_seg += ignore_res.group(0) if ignore_res else word
                    seg_idx += ignore_res.span()[1] if ignore_res else tag_res.span()[1]
                # 需要和后面的tag合并在一起: \\left(  \\left\\{  \\left\vert
                # elif word in ['\\left','\\right']:
                #     ltag_res = re.match(r'\\[A-Za-z]+(\\[A-Za-z]+|\(|\)|\[|\]|\\\{|\\\}|\|)',seg[seg_idx:])
                #     new_seg += colored_word(ltag_res.group(0)) if ltag_res else word
                #     seg_idx += ltag_res.span()[1] if ltag_res else tag_res.span()[1]
                # 保留原样，不加颜色不忽略后文的tag：\it \frac等   
                else:
                    new_seg += word
                    seg_idx += tag_res.span()[1]
                continue
            # invisible符号：_ ^ 等：不加颜色，保留原样
            invisible_res = re.match(r'(\{\[\})|_|\^|\||\{|\}|\$|\\\\|\\$|\[.*\]|&|%|\s+|\*|~|#|\[|\]|(natexlab)|(urlprefix)|\\(!|,|;)',seg[seg_idx:])
            if invisible_res:
                # 上下标没加括号：后面必为一个字母或希腊字母，这时最好加上括号
                if seg[seg_idx] in ['_','^'] and seg_idx < len(seg)-1 and seg[seg_idx+1] != '{':
                    greek_res = re.match(r'\\[A-Za-z]+',seg[seg_idx+1:])
                    if greek_res:   # 希腊字母
                        new_seg += seg[seg_idx]+'{'+greek_res.group(0) + '}'
                        seg_idx += 1+greek_res.span()[1]
                    else:   # 英文字母
                        new_seg += seg[seg_idx]+'{'+seg[seg_idx+1] + '}'
                        seg_idx += 2
                else:
                    new_seg += invisible_res.group(0)  
                    seg_idx += invisible_res.span()[1] 
                continue
            # visible符号：字母、数字、运算符、转义字符等: 加颜色
            visible_res = re.match(r'(\.*\s*[A-Za-z]+)|(\d+(,|\.)*)+|\+|-|/|>|<|=|\(|\)|,|\.|@|;|!|\?|\:|\'|\"|`|(\\[^\\])',seg[seg_idx:])
            if visible_res:
                new_seg += colored_word(visible_res.group(0))
                seg_idx += visible_res.span()[1]
                continue
            
            else: # 理论上不应该到这里，先保留原样
                print(f'unhandled char:"{seg[seg_idx:]}"')
                new_seg += seg[seg_idx]
                seg_idx += 1
                continue
                
         
        texts[j] = new_seg   
        
    color_text = ' '.join(texts)
    return color_text

def colored_file(tex_file):
    
    with open(tex_file,'r',encoding='utf-8') as fi:
        lines = fi.readlines()
    # 删除注释和不必要的定义头
    lines = [line for line in lines if not re.match(r'\s*%|\\vskip|\\vspace',line)]
    
    # 删除不必要的换行:latexindent->文本内部不换行；
    for i,line in enumerate(lines):
        if re.search(r'(?<!\\)%',line): # 行内有非转义%
            lines[i] = line[:re.search(r'(?<!\\)%',line).span()[0]]
        if not line.strip():    # 空行：保留，用于.tex段落换行
            continue
        elif re.match(r'(\\end{.*?})',line.strip()):  # \\end{tag}：结尾换行
            lines[i] = line.strip()+'\n'
        elif re.match(r'\\.*?\{',line):     # 非end的\\tag：开头换行
            if line.startswith('\\date{\\today}'):
                line = line.replace('\\today',random_date())    # 将today()函数换为固定日期，防止mmd和PDF生成的日期不同
            # 参考文献后面不处理
            if '\\begin{thebibliography}' in line:
                break
            lines[i] = '\n'+line.strip()
        else:
            lines[i] = ' '+lines[i].strip()+' '     # 其他：不换行,用空格分割（主要是避免\tag和文本连在一起）
    lines = ''.join(lines).split('\n')
    
    '''
    Now lines be like:
    [
        \\title{Entanglement convertibility for infinite dimensional pure bipartite states},
        \\texttt{ywx20@mails.tsinghua.edu.cn} \\ \\texttt{\{shamao, frankkps, wenswu, yanxia, jtien\}@microsoft.com}, \\\\texttt{zywu@se.cuhk.edu.hk}}
        
        This is pure texts balabala.
        
        \\begin{table}\\begin{tabular}{cc|c|c|c|c|}...\\end{tabular}
        }\\end{table}
        
        \\begin{thebibliography}{99}\\bibitem{teleportation}S.L. and H.J. {\\bf 88};\\bibitem{gussian}Giekde. \\end{thebibliography}
    
    ]
    '''

    
    # 忽略掉tex开头的预定义部分, 在\documentclass和\begin{document}之间插入\usepackage{xcolor}
    begin_idx = 0
    for i,line in enumerate(lines):
        if ('\\begin{document}') in line:
           lines.insert(i,'\\usepackage{xcolor}')
           begin_idx = i+1
           break
    
    for i,line in enumerate(lines[begin_idx:]):
        if line.strip():
            # 参考文献后面不处理，且第一行必须换行，否则编译出错
            if '\\begin{thebibliography}' in line:
                lines[i+begin_idx] = '\n'+line
                break
            # 作者地址内部去除花括号
            elif '\\affiliation' in line:
                content = re.match(r'\\affiliation\{(.*)\}',line[line.index('\\affiliation'):]).group(1)
                content = re.sub(r'\{|\}',' ',content)
                line = line[:line.index('\\affiliation')]+f'\\affiliation{{{content}}}'
            elif '\\def' in line:
                continue
            try:
                lines[i+begin_idx] = colored_text(line)
            except ValueError as e:
                raise 
        # 空行：保留
        
    with open(tex_file.replace('arxiv_origin','arxiv_color'),'w',encoding='utf-8') as fo:
        fo.writelines([line+'\n' for line in lines])
                
def remove_package(tex_file):
    with open(tex_file,'r',encoding='utf-8') as fi:
        lines=fi.read().replace('\\usepackage{xcolor}','%\\usepackage{xcolor}\n')
    lines = lines.replace('\\textcolor','textcolor')
    with open(tex_file,'w',encoding='utf-8') as fo:
        fo.write(lines)
        
def colored_dct(mmd_file):
    # dct: {(r,g,b):'string'}
    dct_color = OrderedDict()
    with open(mmd_file,'r') as fi:
        mmd = fi.read()
    # fix bugs
    mmd = mmd.replace('{t}extcolor','textcolor').replace('_textcolor','textcolor').replace('textcolor','')
    # 去掉多余空格和不统一的格式
    mmd = re.sub(r'\[RGB\]\s*{(.*?)}{(.*?)}',lambda x: x.group(0).replace(' ',''),mmd)
    mmd = re.sub(r'\[\s*RGB\s*]{\s*(\d{3})\s*,\s*(\d{3})\s*,\s*(\d{3})\s*}{(.*?)(?=\[RGB)',r'[RGB]\1,\2,\3\4',mmd)  # 部分地方会缺少右括号
    mmd = re.sub(r'\[\s*RGB\s*]{\s*(\d{3})\s*,\s*(\d{3})\s*,\s*(\d{3})\s*}{(.*?)}',r'[RGB]\1,\2,\3\4',mmd)
    mmds = [mmd for mmd in re.split(r'(?=\[RGB\])|(\n)|\s',mmd) if mmd]
    color = (0,0,0)
    for mmd in mmds:
        res = re.match(r'\[RGB\](\d{3}),(\d{3}),(\d{3})(\S+)',mmd)
        if res:
            color = (int(res.group(1)),int(res.group(2)),int(res.group(3)))
            dct_color[color] = res.group(4)
        # 没匹配上，扔掉脏数据  
        elif 'RGB' in mmd or re.search(r'\d{3},\d{3},\d{3}',mmd):
            continue
        else:   
            # not included：对上一个color进行递增，目的是避免键的重复
            color = add_rgb(color)
            dct_color[color] = mmd
    gt = ' '.join(dct_color.values())
    with open(mmd_file.replace('color.mmd','clean.mmd'),'w') as fo:
        fo.write(gt)
    return dct_color




def match_datapair(mmd_color_dct,pdf_file,png_dir,save_data_path):
    '''
    TODO:
    转为黑白图片
    '''
    pdf = fitz.open(pdf_file)
    
    for page_idx in range(len(pdf)):
        # start a page
        page = pdf[page_idx]
        
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
        ref_lst = []
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
                        # 发生换行时会有两个token对应同一个color，此时只保留第一个position
                        if color not in fitz_color_dct.keys():
                            fitz_color_dct[color] = (text,[[x1,y1],[x2,y2]])
                        elif page_idx >= len(pdf)-2 and color == (0,0,0) and text == 'References':
                            ref_lst.append((text,[[x1,y1],[x2,y2]]))
                                
        pretext = []
        prompt = []
        
        # fitz_color_dct: 当页内容(color升序<=>latex顺序)，mmd_color_dct：全文内容(mmd顺序)
        fitz_color_dct = OrderedDict(sorted(fitz_color_dct.items(),key=lambda x:(x[0][0],x[0][1],x[0][2])))
        if (0,0,0) in fitz_color_dct.keys():
            fitz_color_dct.pop((0,0,0))
       
        # fitz中有但mmd落掉的
        mmd_lost_items = sorted(set(fitz_color_dct.keys())-set(mmd_color_dct.keys()))
        while len(fitz_color_dct) > 0:
            
            # ground_truth: mmd
            mmd_color = next(iter(mmd_color_dct))     
            mmd_text = mmd_color_dct[mmd_color]
           

            if mmd_color in fitz_color_dct.keys():
                # mmd丢了内容：先添加fitz，mmd不动
                # if mmd_lost_items and greater_rgb(mmd_color,mmd_lost_items[0]):
                #     t_and_p = fitz_color_dct[mmd_lost_items[0]]
                #     pretext.append(t_and_p[0])
                #     prompt.append(t_and_p[1])
                #     fitz_color_dct.pop(mmd_color)
                #     mmd_lost_items = mmd_lost_items[1:]
                # 匹配到了，mmd和fitz同时前进

                # else:
                    
                pretext.append(mmd_text)
                prompt.append(fitz_color_dct[mmd_color][1])
                mmd_color_dct.popitem(last=False)
                fitz_color_dct.pop(mmd_color)
            
            else:
                # 非显式符号:需要确保当前段落确实在该页面（图表重排）
                # TODO: 这里Position先作为mask, 实际最好输入下一个词的位置，因为此类大多数为角标引用等
                
                # 找到下一个显式符号
                for c in iter(mmd_color_dct):
                    if not inserted_color(c):
                        if c in fitz_color_dct.keys():
                            # 还是当前页面：先添加mmd,fitz不动
                            pretext.append(mmd_text)
                            prompt.append(['mask'])
                            mmd_color_dct.popitem(last=False)
                        else:
                            # mmd到下一页了，按照fitz遍历

                            # 非第一页的开头，存在不在当前页面的mmd_color，说明是上一页遗留的无效字符
                            if page_idx and len(pretext)<100:
                                mmd_color_dct.pop(mmd_color)
                            fitz_color = next(iter(fitz_color_dct))
                            if fitz_color in mmd_color_dct.keys():
                                pretext.append(mmd_color_dct[fitz_color])
                                mmd_color_dct.pop(fitz_color)
                            else:   # 一般不会走到这里
                                pretext.append(fitz_color_dct[fitz_color][0])
                            prompt.append(fitz_color_dct[fitz_color][1])
                            fitz_color_dct.pop(fitz_color)
                        break
        
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
    '''
    TODO
    affiliation试试匹配，不行的话从pdf上也删掉
    根据彩色pdf的页数判断是否全部成功生成，如果没有还要不要？
    匹配color的时候注意页码是否完整，有无影响？
    最后：re.compile提高效率
    '''
    
    for i,tex_fold in enumerate(fold_lst[start_idx:]):
        break_flag=False
        try:
            os.makedirs(f'data/arxiv_all_files/{tex_fold}', exist_ok=True)
            file_idx = i+start_idx
            # 复制tex文件夹: 保留两份
            if os.system(f'cp -r -n ~/../mnt/data/oss_beijing/zhonghansen/arxiv/latex/{tex_fold} data/arxiv_origin/') \
                or os.system(f'cp -r data/arxiv_origin/{tex_fold} data/arxiv_color/'):
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
                    colored_file(tex_file)
                    os.system(f'cp {tex_file.replace("arxiv_origin","arxiv_color")} data/arxiv_all_files/{tex_fold}/{Path(tex_file).name}')
                except ValueError as e:
                    print(f'{file_idx}. colored_file error: {tex_fold} :{e}')
                    break_flag = True
                    break 
            if not break_flag:
                # 找到主文件
                tex_file = os.popen(f'find data/arxiv_color/{tex_fold} -maxdepth 1 -type f -name "*.tex"').read().split('\n')[0]
                # latexmk: tex->pdf
                os.system(f'latexmk -pdfps -f -g -interaction=nonstopmode -cd data/arxiv_color/{tex_fold} -output-directory=outputs {tex_file} > log/construct_data.txt 2>&1')
                pdf_file = f'data/arxiv_color/{tex_fold}/outputs/{str(Path(tex_file).with_suffix(".pdf").name)}'
                if not os.path.exists(pdf_file):
                    pdf_file = os.popen(f'find data/arxiv_color/{tex_fold}/outputs -maxdepth 1 -type f -name "*.pdf" ').read().split('\n')[0]
                if not os.path.exists(pdf_file):
                    print(f'{file_idx}. tex 2 pdf fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {pdf_file} data/arxiv_all_files/{tex_fold}/{Path(pdf_file).name}')
                    
            if not break_flag:           
                # 去掉/usepackage{xcolor}
                remove_package(tex_file)
                # tex->html
                os.system(f'latexmlc --nocomments --includestyles --dest=data/arxiv_color/{tex_fold}/outputs/color.html --path=data/arxiv_color/{tex_fold} {str(Path(tex_file).name)} > log/construct_data.txt 2>&1')
                html_file = f'data/arxiv_color/{tex_fold}/outputs/color.html'
                if not os.path.exists(html_file):
                    print(f'{file_idx}. tex 2 html fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {html_file} data/arxiv_all_files/{tex_fold}/{Path(html_file).name}')
            if not break_flag:    
                # html->mmd
                os.system(f'python nougat/dataset/parser/html2md.py --html data/arxiv_color/{tex_fold}/outputs/color.html --out data/arxiv_color/{tex_fold}/outputs/color.mmd > log/construct_data.txt 2>&1')
                mmd_file = f'data/arxiv_color/{tex_fold}/outputs/color.mmd'
                if not os.path.exists(mmd_file):
                    print(f'{file_idx}. html 2 mmd fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {mmd_file} data/arxiv_all_files/{tex_fold}/{Path(mmd_file).name}')
                
            if not break_flag:
                # mmd->dct_color
                mmd_file = f'data/arxiv_color/{tex_fold}/outputs/color.mmd'
                dct_color = colored_dct(mmd_file)
                os.system(f'cp {mmd_file.replace("color.mmd","clean.mmd")} data/arxiv_all_files/{tex_fold}/')
                
                # construct datapair
                match_datapair(dct_color,pdf_file,png_dir=f'data/arxiv_train_data/png/{tex_fold}/{tex_fold}',save_data_path='data/arxiv_train_data/train.jsonl')
                print(f'{file_idx}. html 2 mmd fail: {tex_fold}')
            # 删除tex文件夹
            os.system(f'rm -r data/arxiv_origin/{tex_fold}') 
            os.system(f'rm -r data/arxiv_color/{tex_fold}')
        except Exception as e:
            print(f'{file_idx}. {tex_fold}:{e}')
            continue
            

    
if __name__ == '__main__':
    # tex_file = 'data/arxiv_origin/0704.0029/0704.0029.tex'
    # colored_file(tex_file)
   
    # png_dir = 'data/arxiv_train_data/png/0704.0017/0704.0029'
    # dct_color=colored_dct('data/arxiv_all_files/0704.0029/color.mmd')
    # pdf_file='data/arxiv_all_files/0704.0029/source.pdf'
    # save_data_path = 'data/arxiv_train_data/train.jsonl'
    # match_datapair(dct_color,pdf_file,png_dir,save_data_path)
    
    with open('data/pdf_list/latex_dir.txt','r') as fi:
        lines = fi.readlines()
    fold_lst = [line.strip() for line in lines]
    main(fold_lst,start_idx=0)
    