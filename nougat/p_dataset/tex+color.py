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
    day = random.randint(1,28)
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
                elif word in ['\\label','\\begin','\\end','\\includegraphics','\\resizebox','\\cline','\\multicolumn','\\multirow','\\pagestyle','\\email',
                              '\\input','\\bibliographystyle','\\bibliography','\\newcommand','\\usepackage','\\preprint','\\ref','\\url','\\bibitem','\\bibinfo','\\bibnamefont'] \
                                or any(banword in word for banword in ['\\cite','\\ref']):  # \\citer,\\citen,...
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
            # $$
            # formular_res = re.match(r'\$.*?\$',seg[seg_idx:])
            # if formular_res:
            #     new_seg += formular_res.group(0)
            #     seg_idx += formular_res.span()[1]
            #     continue
            
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
        lines = fi.read()
    # 删除注释和不必要的定义头
    lines = re.sub(r'%.*?\n|\\vskip.*?\n|\\vspace.*?\n','\n',lines)
    # 参考文献后面不处理
    if '\\begin{thebibliography}' in lines:
        lines,reference = re.split(r'(?=\\begin\{thebibliography\})', lines)
    else:
        reference = ''
    # 开头的预定义部分不处理
    if '\\begin{document}' in lines:
        predef,lines = re.split(r'(?=\\begin\{document\})',lines)
    else:
        predef = ''
        
    
    # 按照真正的换行位置切分：双斜杠；author和affiliation所在的行单独筛选出来
    lines = re.split(r'\n\n|(?=\\author)|(?=\\affiliation)',lines)
    
    for i,line in enumerate(lines):
        line = line.strip().replace('\n','')
        if '\\def' in line:
            continue
        # 去除affiliation和author内部的花括号，否则可能编译出错
        if '\\affiliation' in line: # F\'{\i}sica -> F\'isica
            content = re.match(r'\\affiliation(?:\[.*?\])*\{(.*)\}',line[line.index('\\affiliation'):]).group(1)
            new_content =re.sub(r'(?=[^\^])\{([a-z])\}(?=[^\$])',r'\1',content)
            line = line.replace(content,new_content)
            lines[i] = line
            continue    # affiliation颜色不变
          
        if '\\author' in line:  #\author{C. Bal\'{a}zs$^{1}$} -> C. Bal\'azs$^{1}$
            content = re.match(r'\\author(?:\[.*?\])*\{(.*)\}',line[line.index('\\author'):]).group(1)
            new_content = re.sub(r'(?=[^\^])\{([a-z])\}(?=[^\$])',r'\1',content)
            line = line.replace(content,new_content)
        
        # 将'\\date{\\today}'函数换为固定日期，防止mmd和PDF生成的日期不同
        if re.search(r'\\date\{.*?\}',line):
            line = line.replace('\\today',random_date())  
        
        
        lines[i] = colored_text(line)
        
        
    lines = '\n\n'.join(lines)
        
    lines = predef + '\\usepackage{xcolor}\n\n'+ lines + reference
    with open(tex_file.replace('arxiv_origin','arxiv_color'),'w',encoding='utf-8') as fo:
        fo.writelines(lines)
                
def remove_package(tex_file):
    with open(tex_file,'r',encoding='utf-8') as fi:
        lines=fi.read().replace('\\usepackage{xcolor}','')
    lines = lines.replace('\\textcolor','textcolor')
    # lines = lines.split('\n')
    # for i,line in enumerate(lines):
    #     if not line.strip():
    #         continue
    #     # 先按照textcolor分开，从而\4可以贪婪匹配，避免匹配不完全
    #     words = [word for word in re.split(r'\\textcolor',line) if word]
    #     for j,word in enumerate(words):
    #         words[j] = re.sub(r'\[RGB\]\{(\d{3}),(\d{3}),(\d{3})\}\{(.*)\}',r' \4T\1\2\3 ',word)
    #     lines[i] = ''.join(words)
    # lines = '\n'.join(lines)
  
    
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

def repair_ref(origin_pdf_file,color_pdf_file):
    
                
    pdf_o = fitz.open(origin_pdf_file)
    pdf_c = fitz.open(color_pdf_file)
    for page_idx in range(len(pdf_c)):
        page_o = pdf_o[page_idx]
        page_c = pdf_c[page_idx]
        origin_text = page_o.get_text()
        color_text = page_c.get_text()
        mis_lst=[]

        spans = [] # flatten to match more
        for b in page_c.get_textpage().extractDICT()['blocks']:
            for l in b['lines']:
                for s in l['spans']:
                    spans.append(s)
        for span_idx,s in enumerate(spans):
            if '?' in s['text']:
                content = s['text']
                pre_content = ''.join([spans[idx]['text'] for idx in range(span_idx-3,span_idx) if idx>=0])
                post_content = ''.join([spans[idx]['text'] for idx in range(span_idx+1,span_idx+3) if idx<len(spans)])
                mis_lst.append((tuple(s['bbox']),s['size'],s['font'],pre_content,content,post_content))
            # for l in b['lines']:
            #     for span_idx,s in enumerate(l['spans']):
            #         if '?' in s['text']:
            #             content = s['text']
            #             pre_content = ''.join([l['spans'][idx]['text'] for idx in range(span_idx-3,span_idx) if 0<=idx])
            #             post_content = ''.join([l['spans'][idx]['text'] for idx in range(span_idx+1,span_idx+3) if idx<len(l['spans'])])
            #             mis_lst.append((tuple(s['bbox']),s['size'],pre_content,content,post_content))
        
        for bbox,size,font,pre_content,content,post_content in mis_lst:
            # 覆盖问号
            page_c.add_redact_annot(bbox)
            # 找到正确文本
            if (not pre_content or len(pre_content.strip())<2 or pre_content not in origin_text) and (not post_content or len(pre_content.strip())<2 or post_content not in origin_text):
                continue
            start_idx = origin_text.index(pre_content)+len(pre_content) if pre_content and len(pre_content.strip())>=2 and pre_content in origin_text else origin_text.index(post_content)-len(content)
            end_idx = origin_text.index(post_content) if post_content and len(pre_content.strip())>=2 and post_content in origin_text else origin_text.index(pre_content)+len(pre_content)+len(content)
            correct_text = re.sub(r'\[|\]','',origin_text[start_idx:end_idx+1]).strip()
            if not re.search(r'\d',correct_text):
                continue
            # 添加正确标号
            new_bbox = (bbox[0],bbox[1]-2,bbox[2]+5 if bbox[2]-bbox[0]<10 else bbox[2],bbox[3]+2)
            page_c.add_redact_annot(new_bbox,correct_text,fontsize=size)
        page_c.apply_redactions()

    pdf_c.save('data/arxiv_all_files/0704.0001/diphoton1.pdf')
    


def match_datapair(mmd_color_dct,pdf_file,png_dir,save_data_path,page_rect=(0,0,672,896)):
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
        fitz_color_dct = {(0,0,0):[]}
        ref_lst = []    # 参考文献
        
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
                        if page_idx == 0 and color == (0,0,0) and len(text)>10 and y2 < page.rect[2]//2:
                            for word in text.split(' '):
                                position = page.search_for(word,clip=s['bbox'])
                                position = [[position[0][0],position[0][1]],[position[0][2],position[0][3]]] if position else ["mask"]
                                fitz_color_dct[(0,0,0)].append((word,position))
                        # Reference
                        if page_idx >= len(pdf)-2 and color == (0,0,0) and text == 'References':
                            ref_lst.append((text,[[x1,y1],[x2,y2]]))
                        # 发生换行时会有两个token对应同一个color，此时只保留第一个position
                        elif color not in fitz_color_dct.keys():
                            fitz_color_dct[color] = (text,[[x1,y1],[x2,y2]])
                                
        pretext = []
        prompt = []
        
        # fitz_color_dct: 当页内容(color升序<=>latex顺序)，mmd_color_dct：全文内容(mmd顺序)
        if page_idx > 0 and (0,0,0) in fitz_color_dct.keys():
            fitz_color_dct.pop((0,0,0))
            fitz_color_dct = sorted(fitz_color_dct.items(),key=lambda x:(x[0][0],x[0][1],x[0][2]))
        fitz_color_dct = OrderedDict(fitz_color_dct)   
      
        while len(fitz_color_dct) > 0:
            
            # ground_truth: mmd
            mmd_color = next(iter(mmd_color_dct))     
            mmd_text = mmd_color_dct[mmd_color]
           

            if mmd_color in fitz_color_dct.keys():
             
                pretext.append(' '+mmd_text)
                prompt.append(fitz_color_dct[mmd_color][1])
                mmd_color_dct.popitem(last=False)
                fitz_color_dct.pop(mmd_color)
            # affiliation
            elif page_idx == 0 and fitz_color_dct[(0,0,0)] and mmd_text == fitz_color_dct[(0,0,0)][0]:
                pretext.append(' '+mmd_text)
                prompt.append(fitz_color_dct[(0,0,0)][0][1])
                mmd_color_dct.popitem(last=False)
                fitz_color_dct[(0,0,0)] = fitz_color_dct[(0,0,0)][1:]
            else:
                # 非显式符号:需要确保当前段落确实在该页面（图表重排）
                # TODO: 这里Position先作为mask, 实际最好输入下一个词的位置，因为此类大多数为角标引用等
                
                # 找到下一个显式符号
                for c in iter(mmd_color_dct):
                    if not inserted_color(c):
                        if c in fitz_color_dct.keys():
                            # 还是当前页面：先添加mmd,fitz不动
                            pretext.append(' '+mmd_text)
                            prompt.append(['mask'])
                            mmd_color_dct.popitem(last=False)
                        else:
                            # mmd到下一页了，按照fitz遍历

                            # 非第一页的开头，存在不在当前页面的mmd_color，说明是上一页遗留的无效字符
                            if page_idx and len(pretext)<100:
                                mmd_color_dct.pop(mmd_color)
                            fitz_color = next(iter(fitz_color_dct))
                            if fitz_color in mmd_color_dct.keys():
                                pretext.append(' '+mmd_color_dct[fitz_color])
                                mmd_color_dct.pop(fitz_color)
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
    '''
    TODO
    根据彩色pdf的页数/mmd的行数判断是否全部成功生成，如果没有还要不要？
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
            # os.system(f'mv  data/arxiv_origin/{tex_fold} data/arxiv_origin/{tex_fold.replace(".",'')}')
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
                color_tex_file = os.popen(f'find data/arxiv_color/{tex_fold} -maxdepth 1 -type f -name "*.tex"').read().split('\n')[0]
                origin_tex_file = os.popen(f'find data/arxiv_origin/{tex_fold} -maxdepth 1 -type f -name "*.tex"').read().split('\n')[0]
                # latexmk: tex->pdf
                # 生成原PDF
                os.system(f'latexmk -pdfps -f -g -interaction=nonstopmode -cd data/arxiv_origin/{tex_fold} {origin_tex_file} > log/construct_data.txt 2>&1') 
                os.system(f'cp {Path(origin_tex_file).with_suffix(".aux")} data/arxiv_color/{tex_fold}/')
                os.system(f'cd data/arxiv_color/{tex_fold}/ && bibtex *.aux > ../../../log/construct_data.txt 2>&1')
                os.system(f'cd data/arxiv_color/{tex_fold}/ && pdflatex -f -interaction=nonstopmode {Path(color_tex_file).name} > ../../../log/construct_data.txt 2>&1')    # > log/construct_data.txt 2>&1
                pdf_file = f'{color_tex_file.replace(".tex",".pdf")}'
                if not os.path.exists(pdf_file):
                    pdf_file = os.popen(f'find data/arxiv_color/{tex_fold}/ -maxdepth 1 -type f -name "*.pdf" ').read().split('\n')[0]
                if not os.path.exists(pdf_file):
                    print(f'{file_idx}. tex 2 pdf fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {pdf_file} data/arxiv_all_files/{tex_fold}/{Path(pdf_file).name}')
                    os.system(f'cp {pdf_file.replace("arxiv_color","arxiv_origin")} data/arxiv_all_files/{tex_fold}/origin.pdf')

                    
            if not break_flag:           
                # 去掉/usepackage{xcolor}
                remove_package(color_tex_file)
                # tex->html
                os.system(f'latexmlc --nocomments --includestyles --dest=data/arxiv_color/{tex_fold}/color.html --path=data/arxiv_color/{tex_fold} {str(Path(color_tex_file).name)} > log/construct_data.txt 2>&1')
                html_file = f'data/arxiv_color/{tex_fold}/color.html'
                if not os.path.exists(html_file):
                    print(f'{file_idx}. tex 2 html fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {html_file} data/arxiv_all_files/{tex_fold}/{Path(html_file).name}')
                    with open(html_file,'r') as fi:
                        lines = fi.readlines()
                    if len(lines)< 20:
                        print(f'{file_idx}. tex 2 html fail: {tex_fold}')
                        break_flag = True
                    
            if not break_flag:    
                # html->mmd
                os.system(f'python nougat/dataset/parser/html2md.py --html data/arxiv_color/{tex_fold}/color.html --out data/arxiv_color/{tex_fold}/color.mmd > log/construct_data.txt 2>&1')
                mmd_file = f'data/arxiv_color/{tex_fold}/color.mmd'
                if not os.path.exists(mmd_file):
                    print(f'{file_idx}. html 2 mmd fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {mmd_file} data/arxiv_all_files/{tex_fold}/{Path(mmd_file).name}')
                
            if not break_flag:
                # mmd->dct_color
                mmd_file = f'data/arxiv_color/{tex_fold}/color.mmd'
                dct_color = colored_dct(mmd_file)
                os.system(f'cp {mmd_file.replace("color.mmd","clean.mmd")} data/arxiv_all_files/{tex_fold}/')
                # 修缮PDF的参考文献
                # repair_ref(pdf_file)
                # construct datapair
                # match_datapair(dct_color,pdf_file,png_dir=f'data/arxiv_train_data/png/{tex_fold}/{tex_fold}',save_data_path='data/arxiv_train_data/train.jsonl')
                print(f'{file_idx}. successful matched: {tex_fold}')
            # 删除tex文件夹
            os.system(f'rm -r data/arxiv_origin/{tex_fold}') 
            os.system(f'rm -r data/arxiv_color/{tex_fold}')
        except Exception as e:
            print(f'{file_idx}. {tex_fold}:{e}')
            continue
            

    
if __name__ == '__main__':
    # repair_ref(origin_pdf_file='data/arxiv_all_files/0704.0001/origin.pdf',color_pdf_file='data/arxiv_all_files/0704.0001/diphoton.pdf')
   
    # png_dir = 'data/arxiv_train_data/png/0704.0008/0704.0008'
    # dct_color=colored_dct('data/arxiv_all_files/0704.0008/color.mmd')
    # pdf_file='data/arxiv_color/0704.0008/outputs/genscalar.pdf'
    # save_data_path = 'data/arxiv_train_data/train.jsonl'
    # match_datapair(dct_color,pdf_file,png_dir,save_data_path)
    
    with open('data/pdf_list/latex_dir.txt','r') as fi:
        lines = fi.readlines()
    fold_lst = [line.strip() for line in lines]
    main(fold_lst,start_idx=0)
    