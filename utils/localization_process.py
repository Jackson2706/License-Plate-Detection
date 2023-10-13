from cv2 import resize, cvtColor, COLOR_BGR2RGB, rectangle, getTextSize, FONT_HERSHEY_SIMPLEX, putText
from matplotlib.pyplot import imshow, axis, show



def sliding_window(img, window_sizes, stride):
    img_height, img_width = img.shape[:2]
    windows = []
    for window_size in window_sizes:
        window_width, window_height = window_size = window_size
        for y_min in range(0, img_height-window_height+1, stride):
            for x_min in range(0, img_width-window_width+1, stride):
                x_max = x_min + window_width
                y_max = y_min + window_height
                windows.append([x_min, y_min, x_max, y_max])
    return windows

def pyramid(img, scale=0.8, min_size=(30,30)):
    acc_scale = 1.0
    pyramid_imgs = [(img, acc_scale)]
    i = 0
    while True:
        acc_scale = acc_scale * scale
        h = int(img.shape[0]*acc_scale)
        w = int(img.shape[1]*acc_scale)
        if h < min_size[0] or w < min_size[0]:
            break
        img = resize(img, (w,h))
        pyramid_imgs.append((img, acc_scale*scale**i))
        i =  i + 1
    return pyramid_imgs

def visualize_bbox(img, bboxes, label_encoder):
    img = cvtColor(img, COLOR_BGR2RGB)
    for box in bboxes:
        x_min, y_min, x_max, y_max, predict_id, conf_score = box
        rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        class_name = label_encoder.inverse_transform([predict_id])[0]
        label = f"{class_name} {conf_score}"
        (w,h), _ = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0,.6, 1)
        rectangle(img, (x_min, y_min-20),(x_min+w, y_min), (0,255,0), -1)
        putText(img, label, (x_min, y_min-5), FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),1) 

    imshow(img)
    axis("off")
    show()

