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
import sys
sys.path.append('/mnt/workspace/sunyu/nougat')
from nougat.dataset.split_md_to_pages import split_markdown

with open('nougat/p_dataset/Greek.json','r') as fi:
    GREEK = json.loads(fi.read())
dct_greek2math = {}
for k,v in GREEK.items():
    dct_greek2math[v] = k

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

def box_distance(box1,box2,alpha=16):
    # |x1-x2|<=50,|y1-y2|<=20
    center1 = ((box1[0][0]+box1[1][0])/2,(box1[0][1]+box1[1][1])/2)
    center2 = ((box2[0][0]+box2[1][0])/2,(box2[0][1]+box2[1][1])/2)
    distance = (center1[0]-center2[0])**2+alpha*(center1[1]-center2[1])**2
    return distance**.5

    
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
                    continue
                # 需要和后面[]*{}内容一起保留原样的tag
                elif word in ['\\label','\\begin','\\end','\\includegraphics','\\resizebox','\\cline','\\multicolumn','\\multirow','\\pagestyle','\\email',
                              '\\input','\\bibliographystyle','\\bibliography','\\newcommand','\\usepackage','\\preprint','\\ref','\\url','\\bibitem','\\bibinfo','\\bibnamefont'] \
                                or any(banword in word for banword in ['\\cite','\\ref','hspace','vspace']):  # \\citep,\\citen,\\hspace*
                    ignore_res = re.match(r'\\[A-Za-z]+(?:\[.*?\])*(?:\{.*?\}){1,2}',seg[seg_idx:])
                    new_seg += ignore_res.group(0) if ignore_res else word
                    seg_idx += ignore_res.span()[1] if ignore_res else tag_res.span()[1]
                    continue
                # 需要和后面合并在一起: \\left(  \\left\\{  \\left\vert \\big( \\Bigg\{
                elif re.match(r'(\\[A-Za-z]+)(\(|\)|\[|\]|\\{|\\}|<|>|\||\\[A-Za-z]+)',seg[seg_idx:]):
                    ltag_res = re.match(r'(\\[A-Za-z]+)(\(|\)|\[|\]|\\{|\\}|<|>|\||\\[A-Za-z]+)',seg[seg_idx:])
                    lword = ltag_res.group(0)
                    if lword in MATH:
                        new_seg += colored_word(ltag_res.group(0)) 
                        seg_idx += ltag_res.span()[1] 
                    else:   # 按照其他tag处理，后面的括号是勿匹配
                        new_seg +=  word
                        seg_idx +=  tag_res.span()[1]
                # 保留原样，不加颜色不忽略后文的tag：\it \frac等   
                else:
                    new_seg += word
                    seg_idx += tag_res.span()[1]
                continue
            # (x,y)：坐标，保持不变
            coord_res = re.match(r'\(-?\d+,-?\d+\)',seg[seg_idx:])
            if coord_res:
                new_seg += coord_res.group(0)
                seg_idx += coord_res.span()[1]
                continue
            # 制表符：删不净：保持不变
            vh_res = re.match(r'-?\d+(pt|mm|bp|cm|em|ex|in)',seg[seg_idx:])
            if vh_res:
                new_seg += vh_res.group(0)
                seg_idx += vh_res.span()[1]
                continue
            # invisible符号：_, ^, { 等：不加颜色，保留原样
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
        line = line.strip().replace('\n',' ')
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
    # 去除多余空行
    lines = re.sub('\n\n+','\n\n',lines)
    with open(tex_file.replace('arxiv_origin','arxiv_color'),'w',encoding='utf-8') as fo:
        fo.writelines(lines)
                
def remove_package(tex_file):
    with open(tex_file,'r',encoding='utf-8') as fi:
        lines=fi.read().replace('\\usepackage{xcolor}','')
    lines = lines.replace('\\textcolor','textcolor')
    with open(tex_file,'w',encoding='utf-8') as fo:
        fo.write(lines)
        
def colored_dct(mmd_file):
    # dct: {(r,g,b):'string'}
    dct_color = OrderedDict()
    with open(mmd_file,'r') as fi:
        mmd = fi.read()
    # fix bugs
    mmd = re.sub(r'\u2062','',mmd)  # 删除多余的不可见Unicode编码
    
    greeks = re.findall(r'[\u0370-\u03FF]+',mmd)    # 将α转为\\alpha
    greeks = set(char for item in greeks for char in item)
    for char in greeks:
        mmd = mmd.replace(f'_{char}_',char).replace(char,dct_greek2math[char])
    
    mmd = mmd.replace('{t}extcolor','textcolor').replace('_textcolor','textcolor').replace('textcolor','')
    # 去掉多余空格和不统一的格式
    mmd = re.sub(r'\[RGB\]\s*{(.*?)}{(.*?)}',lambda x: x.group(0).replace(' ',''),mmd)
    mmd = re.sub(r'\[\s*RGB\s*]{\s*(\d{3})\s*,\s*(\d{3})\s*,\s*(\d{3})\s*}{([^\}]*?)(?=\[RGB)',r'[RGB]\1,\2,\3\4',mmd)  # 部分地方会缺少右括号
    mmd = re.sub(r'\[\s*RGB\s*]{\s*(\d{3})\s*,\s*(\d{3})\s*,\s*(\d{3})\s*}{(.*?)}',r'[RGB]\1,\2,\3\4',mmd)
    mmds = [mmd for mmd in re.split(r'(?=\[RGB\])|(\n)|\s',mmd) if mmd]

    color = (0,0,0)
    for mmd in mmds:
        mmd = re.sub(r'','',mmd)
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
                mis_lst.append((tuple(s['bbox']),s['size'],pre_content,content,post_content))
         
        
        for bbox,size,pre_content,content,post_content in mis_lst:
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
    
def norm_box(box,page_h,page_w):
    x1,y1,x2,y2 = box[0]/page_w,box[1]/page_h,box[2]/page_w,box[3]/page_h
    return [[x1,y1],[x2,y2]]

def match_datapair(mmd_color_dct,pdf_file,page_rect=(0,0,672,896),dis_thres=80):
    '''
    TODO:
    
    训练：png_path在构造训练数据时需改为相对train_data文件夹的相对路径
    训练：mmd_text加空格:暂时都加，tokenize的时候，如果空格+单词是一个token则保留，如果不是一个token则去掉空格（此时大概率是符号）
    '''
        
    pdf = fitz.open(pdf_file)
    os.makedirs(Path(pdf_file).parent/'png',exist_ok=True) 
    save_data_path = Path(pdf_file).parent/'train.jsonl'
    
    for page_idx in range(len(pdf)):
        # start a page
        page = pdf[page_idx]
        _,_,page_w,page_h = page.rect
        # page.set_mediabox(fitz.Rect(page_rect))
        dct = {}
        png_path = Path(pdf_file).parent/f'png/{page_idx}.png'  # data/arxiv_all_files/0710.2897/0.png
        
        # dct['pretext'],dct['prompt']

        fitz_color_dct = {} # {(r,g,b):(t,p)}
        ref_lst = []    # 参考文献
        black_dct = {}  # {text1:[b1,b2],text2:[b3]}
        
        for b in page.get_textpage().extractDICT()['blocks']:
            for l in b['lines']:
                for s in l['spans']:
                    color = fitz.utils.sRGB_to_rgb(s['color'] )
                    color = tuple([math.ceil(c/5)*5 for c in color])
                    text = s['text'] 
                    if not re.search(r'\S',text):   # 无内容的忽略，如' '
                        continue
                   
                    # Reference
                    if page_idx >= len(pdf)-2 and color == (0,0,0) and text in ['References','REFERENCES']:
                        ref_lst.append(('\n\n',['mask']))
                        ref_lst.append((' ##',['mask']))
                        norm_position = norm_box(s['bbox'],page_h,page_w)
                        ref_lst.append((text,norm_position))
                    elif ref_lst:
                        ref_lst.append(('\n',['mask']))
                        for word in text.split(' '):
                            position = page.search_for(word,clip=s['bbox'])
                            norm_position = norm_box(position[0],page_h,page_w) if position else ["mask"]
                            ref_lst.append((' '+word,norm_position))
                    # 黑色
                    elif color == (0,0,0) and len(text.strip())>2:
                        for word in text.split(' '):
                            position = page.search_for(word,clip=s['bbox'])
                            norm_position = norm_box(position[0],page_h,page_w) if position else ["mask"]
                            if word not in black_dct.keys():
                                black_dct[word] = []
                            black_dct[word].append(norm_position)
                    
                    # 发生换行时会有两个token对应同一个color，此时只保留第一个position
                    elif color not in  [(0,0,0),(255,255,255)] and color not in fitz_color_dct.keys():
                        norm_position = norm_box(s['bbox'],page_h,page_w)
                        fitz_color_dct[color] = (text,norm_position)
                            
        pretext = []
        prompt = []
        
        # fitz_color_dct: 当页内容(color升序<=>latex顺序)，mmd_color_dct：全文内容(mmd顺序) 
        fitz_color_dct = sorted(fitz_color_dct.items(),key=lambda x:(x[0][0],x[0][1],x[0][2]))
        fitz_color_dct = OrderedDict(fitz_color_dct)   

        # 一页都没有颜色，后面mmd都匹配不上
        if not fitz_color_dct and not ref_lst:
            break

        while len(fitz_color_dct) > 0 and len(mmd_color_dct)>0:
            
            # ground_truth: mmd
            mmd_color_itr = iter(mmd_color_dct)
            mmd_color = next(mmd_color_itr)     
            mmd_text = mmd_color_dct[mmd_color]
           
            # 颜色匹配上了
            if mmd_color in fitz_color_dct.keys():
                # mmd_text中可能存在空格误分，检查下一个mmd_text是否也属于当前color
                if len(mmd_text) < len(fitz_color_dct[mmd_color][0].strip()): 
                    next_mmd_color = next(mmd_color_itr)
                    if mmd_color_dct[next_mmd_color] in fitz_color_dct[mmd_color][0] and inserted_color(next_mmd_color):    
                        pretext.append(' '+fitz_color_dct[mmd_color][0])
                        mmd_color_dct.popitem(last=False)   # pop两次
                # 正常匹配
                else:
                    pretext.append(' '+mmd_text)
                    
                prompt.append(fitz_color_dct[mmd_color][1])
                mmd_color_dct.popitem(last=False)
                fitz_color_dct.pop(mmd_color)  
                    
            else:
                
                
                # token没加颜色或太小pdf识别不出, 判断当前段落是否在该页面:找到下一个显式符号
                for c in mmd_color_itr:
                    if not inserted_color(c):

                        # 还是当前页面：先添加mmd,fitz不动
                        if c in fitz_color_dct.keys():
                            match_flag = False
                            # 黑色：用距离匹配
                            if mmd_text in black_dct.keys():
                                # len>5，直接匹配
                                if len(mmd_text)> 4 and len(black_dct[mmd_text])==1:
                                    match_flag = True
                                    min_box = black_dct[mmd_text][0]
                                else:
                                    # 找到上一个非mask的框
                                    for prebox in prompt[::-1]:
                                        if len(prebox) == 2:
                                            # 找到距离最小的黑色框
                                            min_distance = dis_thres
                                            min_box = ["mask"]
                                            for b in black_dct[mmd_text]:
                                                neighbor_dis = box_distance(b,prebox)
                                                if neighbor_dis < min_distance:
                                                    min_distance = neighbor_dis
                                                    min_box = b
                                                    match_flag = True                                        
                                            break
                            if match_flag:
                                pretext.append(' '+mmd_text)
                                prompt.append(min_box)
                                black_dct[mmd_text].remove(min_box)
                                if not black_dct[mmd_text]:
                                    black_dct.pop(mmd_text)
                            # 黑色没匹配上
                            else:
                                pretext.append(' '+mmd_text)
                                prompt.append(['mask'])
                            mmd_color_dct.popitem(last=False)
                        
                        # 带颜色的mmd到下一页了，按照当前页fitz遍历
                        else:
                            # 非第一页的开头，存在不在当前页面的mmd_color，说明是上一页遗留的无效字符
                            if page_idx and len(pretext)<100:
                                mmd_color_dct.pop(mmd_color)
                                break

                            fitz_color = next(iter(fitz_color_dct))
                            # 图表重排，当前页非黑色pdf与后面的mmd吻合
                            if fitz_color in mmd_color_dct.keys():
                                pretext.append(' '+mmd_color_dct[fitz_color])
                                prompt.append(fitz_color_dct[fitz_color][1])
                                fitz_color_dct.pop(fitz_color)
                                mmd_color_dct.pop(fitz_color)
                            
                            else:   # 页面结束，省了一些未匹配的fitz
                                fitz_color_dct.pop(fitz_color)
                        break 
                # 后面没有加颜色的word了，break while到结尾
                if inserted_color(c):
                    break
        # 此时len(fitz_color_dct)=0，已跳出循环
        for text,position in ref_lst:    
            pretext.append(text)
            prompt.append(position)
                    
        if len(prompt) > 0 and len(prompt)==len(pretext):
            
                     
            dct['image'] = str(png_path)
            with open(png_path, "wb") as f:
                f.write(page.get_pixmap(dpi=300).pil_tobytes(format="PNG"))       
            img = Image.open(png_path)
            bi_img = img.point(lambda pix: 0 if pix%5 else pix)
            bi_img.save(png_path)
            
            dct["label"] = []
            dct["pretext"] = pretext
            dct["prompt"] = prompt
            
            
            
  
            
            with open(save_data_path,'a') as fo:
                json.dump(dct,fo)
                fo.write('\n')
            
           
    if os.path.exists(png_path.parent) and os.path.exists(save_data_path):
        tex_fold = Path(pdf_file).parent.name
        os.system(f'cp -r {png_path.parent} data/arxiv_all_files/{tex_fold}/')
        os.system(f'cp {save_data_path} data/arxiv_all_files/{tex_fold}/')
            
            
def black_datapair(pdf_file,mmd_file):
    pdf = fitz.open(pdf_file)
    os.makedirs(Path(pdf_file).parent/'png',exist_ok=True) 
    save_data_path = Path(pdf_file).parent/'train_black.jsonl'
    with open(mmd_file,'r', encoding="utf-8") as fi:
        mds = fi.read().replace("\xa0", " ")
    mds, meta = split_markdown(mds, pdf)
    
    for page_idx in range(len(pdf)):
        # start a page
        page = pdf[page_idx]
        md = mds[page_idx]
        if md:
            dct = {}

            png_path = Path(pdf_file).parent/f'png/{page_idx}.png'  # data/arxiv_all_files/0710.2897/0.png
            dct['image'] = str(png_path)
            with open(png_path, "wb") as f:
                f.write(page.get_pixmap(dpi=300).pil_tobytes(format="PNG"))       
                
            dct["label"] = []
            dct["pretext"] = md
            dct["prompt"] = ['mask']
                
            with open(save_data_path,'a') as fo:
                json.dump(dct,fo)
                fo.write('\n')
            
           
    if os.path.exists(png_path.parent) and os.path.exists(save_data_path):
        tex_fold = Path(pdf_file).parent.name
        os.system(f'cp -r {png_path.parent} data/arxiv_all_files/{tex_fold}/')
        os.system(f'cp {save_data_path} data/arxiv_all_files/{tex_fold}/')
            

def main(fold_lst,start_idx):
  
    
    for i,tex_fold in enumerate(fold_lst[start_idx:]):
        break_flag=False
        try:
            os.makedirs(f'data/arxiv_all_files/{tex_fold}', exist_ok=True)
            file_idx = i+start_idx
            # 复制tex文件夹: 保留两份
            if os.system(f'cp -r ~/../mnt/data/oss_beijing/zhonghansen/arxiv/latex/{tex_fold} data/arxiv_origin/') \
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
                # 生成原PDF,用.aux和.bib生成彩色pdf
                os.system(f'latexmk -pdfps -f -g -interaction=nonstopmode -cd data/arxiv_origin/{tex_fold} {origin_tex_file} > log/construct_data.txt 2>&1') 
                origin_pdf_file = f'{origin_tex_file.replace(".tex",".pdf")}'
                if not os.path.exists(origin_pdf_file):
                    print(f'{file_idx}. origin 2 pdf fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {origin_pdf_file} data/arxiv_all_files/{tex_fold}/origin.pdf')
                    os.system(f'cp {Path(origin_tex_file).with_suffix(".aux")} data/arxiv_color/{tex_fold}/')
                    os.system(f'cd data/arxiv_color/{tex_fold}/ && bibtex *.aux > ../../../log/construct_data.txt 2>&1')
                    os.system(f'cd data/arxiv_color/{tex_fold}/ && pdflatex -f -interaction=nonstopmode {Path(color_tex_file).name} > ../../../log/construct_data.txt 2>&1')    # > log/construct_data.txt 2>&1
                    color_pdf_file = f'{color_tex_file.replace(".tex",".pdf")}'
                    if not os.path.exists(color_pdf_file):
                        color_pdf_file = os.popen(f'find data/arxiv_color/{tex_fold}/ -maxdepth 1 -type f -name "*.pdf" ').read().split('\n')[0]
                    if not os.path.exists(color_pdf_file):
                        print(f'{file_idx}. tex 2 pdf fail: {tex_fold}')
                        break_flag = True
                    else:
                        os.system(f'cp {color_pdf_file} data/arxiv_all_files/{tex_fold}/{Path(color_pdf_file).name}')                    
                    
            if not break_flag:           
                # 去掉/usepackage{xcolor}
                remove_package(color_tex_file)
                # tex->html
                os.system(f'latexmlc --nocomments --includestyles --dest=data/arxiv_color/{tex_fold}/color.html --path=data/arxiv_color/{tex_fold} -log=data/arxiv_color/{tex_fold}/color.latexml.log {str(Path(color_tex_file).name)} > log/construct_data.txt 2>&1')
                html_file = f'data/arxiv_color/{tex_fold}/color.html'
                if not os.path.exists(html_file):
                    print(f'{file_idx}. tex 2 html fail: {tex_fold}')
                    break_flag = True
                else:
                    os.system(f'cp {html_file} data/arxiv_all_files/{tex_fold}/{Path(html_file).name}')
                    with open(html_file,'r') as fi:
                        lines = fi.readlines()
                    if len(lines)< 40:
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
                dct_color = colored_dct(mmd_file)
                os.system(f'cp {mmd_file.replace("color.mmd","clean.mmd")} data/arxiv_all_files/{tex_fold}/')
                # construct datapair
                match_datapair(dct_color,color_pdf_file)
                print(f'{file_idx}. successful constructed: {tex_fold}')
            # 失败了，生成黑白数据
            elif os.path.exists(origin_pdf_file):
                # origin.html
                os.system(f'latexmlc --nocomments --includestyles --dest=data/arxiv_origin/{tex_fold}/origin.html --path=data/arxiv_origin/{tex_fold} -log=data/arxiv_origin/{tex_fold}/origin.latexml.log {str(Path(origin_tex_file).name)} > log/construct_data.txt 2>&1')
                html_file = f'data/arxiv_origin/{tex_fold}/origin.html'
                if os.path.exists(html_file):
                    os.system(f'cp {html_file} data/arxiv_all_files/{tex_fold}/{Path(html_file).name}')
                    # origin.mmd
                    os.system(f'python nougat/dataset/parser/html2md.py --html data/arxiv_origin/{tex_fold}/origin.html --out data/arxiv_origin/{tex_fold}/origin.mmd > log/construct_data.txt 2>&1')
                    mmd_file = f'data/arxiv_origin/{tex_fold}/origin.mmd'
                    if os.path.exists(mmd_file):
                        os.system(f'cp {mmd_file} data/arxiv_all_files/{tex_fold}/{Path(mmd_file).name}')                     
                        # construct datapair
                        black_datapair(origin_pdf_file,mmd_file)
                        print(f'{file_idx}. black constructed: {tex_fold}')

        except Exception as e:
            print(f'{file_idx}. {tex_fold}:{e}')
            
        finally:
            # 删除tex文件夹
            os.system(f'rm -r data/arxiv_origin/{tex_fold}') 
            os.system(f'rm -r data/arxiv_color/{tex_fold}')
            
    
if __name__ == '__main__':
    # repair_ref(origin_pdf_file='data/arxiv_all_files/0704.0001/origin.pdf',color_pdf_file='data/arxiv_all_files/0704.0001/diphoton.pdf')
   
  
    # dct_color=colored_dct('data/arxiv_all_files/0704.0017/color.mmd')
    # pdf_file='data/arxiv_all_files/0704.0017/source.pdf'

    # match_datapair(dct_color,pdf_file)
    
    with open('data/pdf_list/latex_dir.txt','r') as fi:
        lines = fi.readlines()
    fold_lst = [line.strip() for line in lines]
    main(fold_lst,start_idx=134)
    