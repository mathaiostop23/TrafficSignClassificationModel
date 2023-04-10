import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


np.random.seed(42)


data_dir = '/Users/mathaios/Desktop/python tensroflow'
train_path = '/Users/mathaios/Desktop/python tensroflow/Train'
test_path = '/Users/mathaios/Desktop/python tensroflow/Test'


channels = 3

classes = len(os.listdir(train_path))

class_names = np.array(['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'])

folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(class_names[int(folder)])
    
    
    
image_data = []
image_labels = []

for i in range(classes):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)
    for img in images:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((30, 30))
            image_data.append(np.array(resize_image))
            image_labels.append(i)


image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255

y_train = keras.utils.to_categorical(y_train, classes)
y_val = keras.utils.to_categorical(y_val, classes)

print(y_train.shape)
print(y_val.shape)



model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))




model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    fill_mode="constant")

history = model.fit(aug.flow(X_train, y_train, batch_size=64), epochs=25, validation_data=(X_val, y_val))



test = pd.read_csv('/Users/mathaios/Desktop/python/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((30, 30))
        data.append(np.array(resize_image))
        
X_test = np.array(data)
X_test = X_test/255

pred_prob = model.predict(X_test)
pred_classes = np.argmax(pred_prob, axis=1)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(labels, pred_classes)*100)

plt.figure(figsize = (7, 7))

start_index = 0
for i in range(40):
    plt.subplot(8, 6, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred_classes[start_index + i]
    actual = labels[start_index + i]
    col = 'g'
    if (prediction != actual).all():
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(class_names[actual], class_names[prediction]), color = col)
    plt.imshow(X_test[start_index + i])
plt.show()


model.save('TrafficSign.h5')