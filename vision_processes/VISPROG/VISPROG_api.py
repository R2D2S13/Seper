from vision_processes.vision_processes import forward

def forward_wrap(model_name, *args,queues=None,**kwargs):
    return forward(model_name, *args, queues=queues, **kwargs)

#model:owlvit-large-patch14
#name: owlvit_visprog
def LOC(image,object,queues=None):
    w,h = image.size   
    if object=='TOP':
        bbox = [[0,0,w-1,int(h/2)]]
    elif object=='BOTTOM':
        bbox = [[0,int(h/2),w-1,h-1]]
    elif object=='LEFT':
        bbox = [[0,0,int(w/2),h-1]]
    elif object=='RIGHT':
        bbox = [[int(w/2),0,w-1,h-1]]
    else:
        bbox = forward_wrap('owlvit_visprog',image,object,queues=queues)

    return bbox

#model: blip-vqa-capfilt-large
#name: 'blip1_visprog'
def VQA(image,question,queues=None):
    answer = forward_wrap('blip1_visprog',image,question,queues=queues)
    return answer

def RESULT(var):
    return var

def EVAL(expr):
    if 'xor' in expr:
        expr = expr.replace('xor','!=')
    return eval(expr)

def COUNT(box):
    return len(box)

def expand_box(box,image_size,factor=1.5):
    W,H = image_size
    x1,y1,x2,y2 = box
    dw = int(factor*(x2-x1)/2)
    dh = int(factor*(y2-y1)/2)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    x1 = max(0,cx - dw)
    x2 = min(cx + dw,W)
    y1 = max(0,cy - dh)
    y2 = min(cy + dh,H)
    return [x1,y1,x2,y2]

def CROP(image,box):
    if len(box) > 0:
        box = box[0]
        box = expand_box(box, image.size)
        out_image = image.crop(box)
    else:
        box = []
        out_image = image

    return out_image

def CROP_RIGHTOF(image,box):
    if len(box) > 0:
        box = box[0]
        w,h = image.size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        right_box = [cx,0,w-1,h-1]
    else:
        w,h = image.size
        box = []
        right_box = [int(w/2),0,w-1,h-1]

    out_image = image.crop(right_box)
    
    return out_image

def CROP_LEFTOF(image,box):
    if len(box) > 0:
        box = box[0]
        w,h = image.size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        left_box = [0,0,cx,h-1]
    else:
        w,h = image.size
        box = []
        left_box = [0,0,int(w/2),h-1]
    
    out_image = image.crop(left_box)


    return out_image

def CROP_BELOW(image,box):
    if len(box) > 0:
        box = box[0]
        w,h = image.size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        below_box = [0,cy,w-1,h-1]
    else:
        w,h = image.size
        box = []
        below_box = [0,0,int(w/2),h-1]
    
    out_image = image.crop(below_box)

    return out_image

def CROP_ABOVE(image,box):
    if len(box) > 0:
        box = box[0]
        w,h = image.size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        above_box = [0,0,w-1,cy]
    else:
        w,h = image.size
        box = []
        above_box = [0,0,int(w/2),h-1]
    
    out_image = image.crop(above_box)


    return out_image

def CROP_FRONTOF(image,box):
    return CROP(image,box)

def CROP_INFRONT(image,box):
    return CROP(image,box)

def CROP_INFRONTOF(image,box):
    return CROP(image,box)

def CROP_BEHIND(image,box):
    return CROP(image,box)

def CROP_AHEAD(image,box):
    return CROP(image,box)

