from PIL import Image,ImageOps,ImageDraw
import torch

def visual_box(png_path,boxes,save_path,color='red',image_size = [672,896]):
    img = Image.open(png_path).resize(image_size)
    draw = ImageDraw.Draw(img)
    boxes = boxes.reshape(-1,2,2)
    for box in boxes:
        box[:,0] *= image_size[0]
        box[:,1] *= image_size[1]
        draw.rectangle([tuple(box[0]),tuple(box[1])],outline=color)
        # draw.text((box[0][0]-10,box[0][1]-20),f'{box.tolist()}',fill='black')
    img.save(save_path)   