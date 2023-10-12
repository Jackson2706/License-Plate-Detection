from sklearn.preprocessing import LabelEncoder
from cv2 import imread, imwrite

from Dataset.dataset import Dataset
from Utils.preprocess import preprocess_img, crop_object

"""
    Preprocessing dataset before training SVM
"""

train_dataset = Dataset(dataset_dir="License_Plate-5", phase="train")
image_path_list, annotation_list = train_dataset.__call__()
for image_path, annotation in zip(image_path_list, annotation_list):
    [[x1,y1,x2,y2, label]] = annotation
    image = imread(str(image_path))
    imwrite("original.jpg", image)
    object_img = crop_object(image, [x1,y1,x2,y2])
    imwrite("LP.jpg", object_img)
    hog_object_image = preprocess_img(image)
    print(hog_object_image.shape)
    imwrite("test.jpg", hog_object_image)
    
