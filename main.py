from cv2 import imread
from joblib import load
from numpy import argmax
import numpy as np
from utils.localization_process import pyramid, sliding_window, nms
from utils.classification_preprocess import preprocess_img
from utils.metrics import visualize_bbox
from time import time


start = time()
image_path = "License_Plate-5/test/5587_jpg.rf.026eb4698e8035d73abea983bb0ef785.jpg"
window_size = [
    (200,50),
    (250,40)
]
stride = 12
conf_threshold = 0.99
iou_threshold = 0.05
image = imread(image_path)

clf, scaler, label_encoder = load("weights/clf_model_and_scaler_feature.pkl")

pyramid_imgs = pyramid(image)
bboxes = []
for pyramid_img_info in pyramid_imgs:
    pyramid_img, scale_factor = pyramid_img_info
    window_lst = sliding_window(pyramid_img,
                                window_sizes=window_size,
                                stride=stride
                                )
    for window in window_lst:
        xmin, ymin, xmax, ymax = window
        object_img = pyramid_img[ymin:ymax, xmin:xmax]
        preprocessed_img = preprocess_img(object_img)
        normalized_img = scaler.transform([preprocessed_img])[0]
        decision = clf.predict_proba([normalized_img])[0]
        if np.all(decision < conf_threshold):
            continue
        predict_id = argmax(decision)
        conf_score = decision[predict_id]
        xmin  = int(xmin / scale_factor)
        ymin = int(ymin / scale_factor)
        xmax = int(xmax / scale_factor)
        ymax = int(ymax / scale_factor)
        bboxes.append([xmin, ymin, xmax, ymax, predict_id, conf_score])

bboxes = nms(bboxes, iou_threshold=iou_threshold)
lp_bbox = [bbox for bbox in bboxes if bbox[4] == 0]
print(lp_bbox)
end = time()
print("Duration: {}".format(end-start))
visualize_bbox(image, lp_bbox, label_encoder)
