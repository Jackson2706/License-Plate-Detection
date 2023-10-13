from cv2 import imread
from joblib import load
from numpy import argmax, all
from utils.localization_process import pyramid, sliding_window
from utils.classification_preprocess import preprocess_img

image_path = "5587_jpg.rf.026eb4698e8035d73abea983bb0ef785.jpg"
window_size = [
    (32,32),
    (64,64),
    (128,128)
]
stride = 12
conf_threshold = 0.8
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
        if all(decision < conf_threshold):
            continue
        predict_id = argmax(decision)
        conf_score = decision[predict_id]
        bboxes.append([xmin, ymin, xmax, ymax, predict_id, conf_score])
        
