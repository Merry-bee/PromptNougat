
import os
from pathlib import Path
import argparse

import re
import fitz
from markdown import markdown



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_dir", type=str, required=True)
    parser.add_argument("--pdf_dir", type=str, required=True)
    parser.add_argument("--png_dir", type=str, required=True)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    # python -m pdb nougat/dataset/save_error_png.py --error_dir output/greedy_search/error --pdf_dir data/quantum --png_dir data/train_data/png2 --save True
    image_size=[896,672]

    for error_mmd_file in os.listdir(args.error_dir):
        error_mmd_path = os.path.join(args.error_dir,error_mmd_file)
        pdf_path = os.popen(f'find {args.pdf_dir} -name {Path(error_mmd_file).with_suffix(".pdf")}').read().strip()
        with open(error_mmd_path,'r') as fi:
            lines = fi.read()
        missing_pages = re.findall(r'MISSING_PAGE_FAIL:\d+.*?MISSING_PAGE_FAIL:\d+',lines,re.S)
        pdf = fitz.open(pdf_path)
        for missing_page in missing_pages:
            if missing_page[:19]!=missing_page[-19:]:
                continue    # 不是一页，匹配错了
            page_num = int(re.match(r'MISSING_PAGE_FAIL:(\d+)',missing_page,re.S).group(1))
            mat = fitz.Matrix(image_size[1]/pdf[page_num-1].rect.width, image_size[0]/pdf[page_num-1].rect.height)
            png_path = os.path.join(args.png_dir, f'{Path(pdf_path).stem}_{page_num}.png')
            with open(png_path, "wb") as f:
                f.write(pdf[page_num-1].get_pixmap(matrix=mat).pil_tobytes(format="PNG"))
            txt_path = Path(png_path).with_suffix('.txt')
            with open(txt_path,'w') as fi:
                fi.writelines(missing_page)
                