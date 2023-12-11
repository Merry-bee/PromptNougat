import pylatexenc

def main(tex_file):
    with open(tex_file,'r',encoding='utf-8') as fi:
        tex = fi.read()
    decoded_tex = pylatexenc.latex2text(tex)
   
if __name__ == '__main__':
    main('data/tmp/2110.07274/main.tex')
    