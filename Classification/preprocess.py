from cv2 import imread, resize, 
from numpy import float64
from skimage.transform import resize
from skimage.features import hog

"""
    Extract images from RGB images to 1D images with using HoG method
    @param: image_path, for example: "./images/abc.jpg"
    @return: a hoG image

"""
def preprocess_img(img_path):
    img = imread(img_path)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float64)
    resized_img = resize(img, output_shape=(32,32), anti_aliasing= True) 
    hog_features = hog(resized_img, orientation=9, pixels_per_pixel=(8,8), cell_per_block=(2,2),
                       transform_sqrt=True, block_norm="L2", feature_vector=True)
    return hog_features


def crop_object(img, bbox):
    x_min, y_min, x_max, y_max = bbox
    object_image = img[y_min:y_max, x_min:x_max]
    return object_image