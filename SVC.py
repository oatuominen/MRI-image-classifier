import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from skimage.transform import resize 
from skimage.io import imread 
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss


data_dir = 'archive'   # training and validation data directory
test_data_dir = 'test' # test data directory

# Setting image size and class names
img_height, img_width = 128, 128
classes = ['healthy','tumor']

# Function for loading data from a directory
def load_data(data_dir, classes, img_height, img_width):
    data_arr = []   # stores image data
    target_arr = [] # stores labels
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # looping through classes ('healthy', 'tumor')
    for i in classes:
        path = os.path.join(data_dir, i)
        # looping through each image in the class
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.splitext(img)[-1].lower() in valid_extensions:
                img_array = imread(img_path)
                img_resized = resize(img_array, (img_width, img_height))
                data_arr.append(img_resized.flatten())  
                target_arr.append(classes.index(i)) # Append index of the class (0: 'healthy', 1: 'tumor')
    
    # Converting to numpy arrays
    data = np.array(data_arr)
    target = np.array(target_arr)
    
    return data, target


# Load training and test data:

# Using the load_data function
train_data, train_labels = load_data(data_dir, classes, img_height, img_width)
test_data, test_labels = load_data(test_data_dir, classes, img_height, img_width)



# Splitting the training data into training and validation sets (80-20 split)
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, 
                                                  random_state=77, stratify=train_labels)



# Initializing the SVC model
svc = SVC(kernel='rbf', class_weight='balanced')


# Training the model
svc.fit(x_train, y_train)

# Predicting labels for the training set
y_train_pred = svc.predict(x_train)

# Calculating training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_train_decision = svc.decision_function(x_train)
y_train_mod = np.where(y_train == 1, 1, -1)
train_hinge_loss = hinge_loss(y_train_mod, y_train_decision)
print(f"Training Hinge Loss: {train_hinge_loss:.4f}")

# Predicting labels for the validation set
y_val_pred = svc.predict(x_val)


# Calculating validation accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


y_val_decision = svc.decision_function(x_val)
y_val_mod = np.where(y_val == 1, 1, -1)
val_hinge_loss = hinge_loss(y_val_mod, y_val_decision)
print(f"Validation Hinge Loss: {val_hinge_loss:.4f}")

# Predicting labels for the test set
y_test_pred = svc.predict(test_data)

# Calculating test accuracy
test_accuracy = accuracy_score(test_labels, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


y_test_decision = svc.decision_function(test_data)
y_test_mod = np.where(test_labels == 1, 1, -1)
test_hinge_loss = hinge_loss(y_test_mod, y_test_decision)
print(f"Test Hinge Loss: {test_hinge_loss:.4f}")


