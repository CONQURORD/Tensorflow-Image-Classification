import keras.layers
from tensorflow.keras import *
import numpy as np
from sklearn.feature_extraction import image
import pandas as pd

df = pd.read_csv('data_csvs_out_basic.csv')

X = df.iloc[:,2:]

y = df.iloc[:,1:2]



import matplotlib.pyplot as plt

#print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

class_names = ['Up','Down']

#print(X_train.shape)
#print(y_train)
print(X_test.shape)
#print(y_test)

X_train = X_train / 255.0

X_test = X_test / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.imshow(X_train[i],cmap=plt.cm.binary)
     plt.grid(False)
     plt.xlabel(class_names[y_train[i]])
#plt.show()

model = keras.Sequential([
     keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(128,activation='relu'),
     keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5)

test_loss, test_acc = model.evaluate(X_test,y_test)

print("Test Accuracy: {} ".format(test_acc))
print("Test Loss: {} ".format(test_loss))

pred = model.predict(X_test)

print(pred[0].shape)