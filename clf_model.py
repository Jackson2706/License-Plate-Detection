from sklearn.preprocessing import LabelEncoder, StandardScaler
from cv2 import imread, imwrite
from numpy import array
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Dataset.dataset import Dataset
from Utils.preprocess import preprocess_img, crop_object

"""
    Preprocessing dataset before training SVM
"""
def preprocess_dataset(image_path_list, annotation_list):
    image_feature_list = []
    label_list = []
    for image_path, annotation in zip(image_path_list, annotation_list):
        [[x1,y1,x2,y2, label]] = annotation
        image = imread(image_path)
        object_img = crop_object(image, [x1,y1,x2,y2])
        hog_object_image = preprocess_img(object_img)
        image_feature_list.append(hog_object_image)
        label_list.append(label)
    return array(image_feature_list), array(label_list)

if __name__ == '__main__':
    train_dataset = Dataset(dataset_dir="License_Plate-5", phase="train")
    train_image_path_list, train_annotation_list = train_dataset.__call__()

    val_dataset = Dataset(dataset_dir="License_Plate-5", phase="valid")
    val_image_path_list, val_annotation_list = val_dataset.__call__()

    '''
        Training phase
    '''
    X_train, y_train = preprocess_dataset(image_path_list=train_image_path_list, annotation_list=train_annotation_list)
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # Normalize the feature
    scaler = StandardScaler()
    scaler.fit_transform(X_train)

    # Defining the model
    clf = SVC(
        kernel='rbf',
        random_state = 123,
        probability = True,
        C=0.5
    )
    clf.fit(X_train, y_train)

    '''
        Validation phase
    '''

    X_val, y_val = preprocess_dataset(image_path_list= val_image_path_list, annotation_list=val_annotation_list)
    # Encode the labels
    y_val = label_encoder.transform(y_val)
    # Normalize the features
    scaler.transform(X_val)

    # Predict val data
    y_pred = clf.predict(X_val)
    # Evaluate the model
    acc_score = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    print("Accuracy: {} \nPrecision: {} \nRecall: {} \n".format(acc_score, precision, recall))


    
