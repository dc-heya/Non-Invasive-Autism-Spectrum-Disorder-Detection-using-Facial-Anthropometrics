
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import random
import os
import gc

from tensorflow import keras
from keras import layers
from keras import models, Sequential
from keras import optimizers
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


# In[128]:


# Creating file path for our train data and test data
train_dir = "AutismDataset/train"
test_dir = "AutismDataset/test"


# In[129]:


# Getting 'Autistic' and 'Non-Autistic' train images from respective file names of train data
train_non_autistic = []
train_autistic = []
for i in os.listdir(train_dir):
    if 'Non_Autistic' in ("AutismDataset/train/{}".format(i)):
        train_non_autistic.append(("AutismDataset/train/{}".format(i)))
    else:
        train_autistic.append(("AutismDataset/train/{}".format(i)))

# Getting test images from test data file path
test_imgs = ["AutismDataset/test/{}".format(i) for i in os.listdir(test_dir)]


# Concatenate 'Autistic'  and 'Non-Autistic' images and shuffle them as train_images
train_imgs = train_autistic + train_non_autistic
random.shuffle(train_imgs)

# Remove the lists to save space
del train_autistic
del train_non_autistic
gc.collect()
     


# In[130]:


# Set the dimensions for images
nrows = 150
ncolumns  = 150
channels = 3

# Read and process the images: Function returns X,y. X - list of resized images, y - list of labels for the images

def read_and_process_image(list_of_images):
    X = []
    y = []

    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation = cv2.INTER_CUBIC))
        if 'Non_Autistic' in image:
            y.append(0)
        else:
            y.append(1)

    return X,y


# In[131]:


# Get resized images and labels from train data
X_train, y_train = read_and_process_image(train_imgs)

# Delete train images to save space
del train_imgs
gc.collect()


# In[132]:


# Convert the lists to array
plt.figure(figsize=(12, 8))
X_train = np.array(X_train)
y_train = np.array(y_train)
sns.countplot(y_train, saturation=1)
plt.title("Train image labels")


# In[133]:


# Shape of train images and labels
print("Shape of train images:", X_train.shape)
print("Shape of train labels:", y_train.shape)


# In[134]:


# Repeat the above process for validation data to get val_images
val_autistic = "AutismDataset/valid/Autistic"
val_non_autistic = "AutismDataset/valid/Non_Autistic"
val_autistic_imgs = ["AutismDataset/valid/Autistic/{}".format(i) for i in os.listdir(val_autistic)]
val_non_autistic_imgs = ["AutismDataset/valid/Non_Autistic/{}".format(i) for i in os.listdir(val_non_autistic)]
val_imgs = val_autistic_imgs + val_non_autistic_imgs
random.shuffle(val_imgs)

# Remove the lists to save space
del val_autistic_imgs
del val_non_autistic_imgs
gc.collect()


# In[135]:


# Get resized images and labels from validation data
X_val, y_val = read_and_process_image(val_imgs)

# Delete validation images to save space
del val_imgs
gc.collect()


# In[136]:


# Convert the lists to array
plt.figure(figsize=(12, 8))
X_val = np.array(X_val)
y_val = np.array(y_val)
sns.countplot(y_val, saturation=1)
plt.title("Validation image labels")


# In[137]:


# Shape of validation images and labels
print("Shape of validation images:", X_val.shape)
print("Shape of validation labels:", y_val.shape)


# In[138]:


# Get length of train data and validation data
ntrain = len(X_train)
nval = len(X_val)
batch_size = 32


# #VGG16 Model Implementation

# In[13]:


# Calling pre-trained VGG16 model
base_model = VGG16(include_top=False,weights='imagenet',input_shape=(150,150,3))


# In[14]:


#vgg16
print("Number of layers in the base model: ", len(base_model.layers))
     


# In[15]:


# Freeze the layers in pre-trained model, we don't need to train again
for layer in base_model.layers:
   layer.trainable = False


# In[16]:


# Create our classifier model, connect pre-trained model vgg to our model
model = keras.models.Sequential()
model.add(base_model)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))


# In[17]:


# Create summary of our model
model.summary()


# In[18]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[19]:


# Configure data augumentation and scaling of images to prevent overfitting since we have a small train data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# Only rescaling for validation data
val_datagen = ImageDataGenerator(rescale = 1./255)
     


# In[20]:


X_test, y_test = read_and_process_image(test_imgs)
del test_imgs
gc.collect()


# In[21]:


plt.figure(figsize=(12, 8))
X_test = np.array(X_test)
y_test = np.array(y_test)
sns.countplot(y_val, saturation=1)
plt.title("Test image labels")
     


# In[22]:


# Create test and validation image generator
BATCH_SIZE = 64
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)
train_generator = train_datagen.flow(X_train, y_train, batch_size = BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size = BATCH_SIZE)


# In[86]:


# Train the model
#early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Entrenar el modelo
history = model.fit(train_generator,
                              epochs=40,
                              validation_data=val_generator,
                              #callbacks=[early_stopping],
                              workers=8,
                              use_multiprocessing=False
                             )


# In[87]:


# Learning curves for training and validation
history_df = pd.DataFrame(history.history)
history_df
     


# In[88]:


plt.figure(figsize=(12, 8))
sns.lineplot(data=history_df.loc[:, ["accuracy", "val_accuracy"]], palette=['b', 'r'], dashes=False)
sns.set_style("whitegrid")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
     


# In[89]:


X = np.array(X_test)


# In[90]:


pred = model.predict(X)
threshold = 0.5
predictions = np.where(pred > threshold, 1,0)


# In[103]:


test = pd.DataFrame(data = predictions, columns = ["predictions"])
test
test["filename"] = [os.path.basename(i) for i in test_imgs]
test["test_labels"] = y_test
test = test[["filename", "test_labels", "predictions"]]
test
     


# In[104]:


model_accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: {:.2f}%".format(model_accuracy * 100))


# In[105]:


cl_report = classification_report(y_test, predictions)
print(cl_report)


# In[106]:


cn_matrix= confusion_matrix(y_test, predictions)
cn_matrix


# In[107]:


f, ax = plt.subplots(figsize = (8,6))
ax = sns.heatmap(cn_matrix, annot=True,fmt="d")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")


# #Xception Model Implementation 

# In[45]:


# Calling pre-trained xception model
base_model1 = Xception(include_top=False,weights='imagenet',input_shape=(150,150,3),pooling='avg')


# In[46]:


# Let's see how many layers are in the xception model
print("Number of layers in the base model: ", len(base_model1.layers))


# In[47]:


# Freeze the layers in pre-trained model, we don't need to train again
for layer in base_model1.layers:
   layer.trainable = False


# In[48]:


model1 = Sequential()
model1.add(base_model1)
model1.add(layers.Dense(512, activation = 'relu'))
model1.add(layers.Dense(units = 256 , activation = 'relu'))
model1.add(Dense(units = 64 , activation = 'relu'))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1, activation = 'sigmoid'))


# In[49]:


# Create summary of our model
model1.summary()


# In[50]:


model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[51]:


#  sammeeeeeeee Configure data augumentation and scaling of images to prevent overfitting since we have a small train data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# Only rescaling for validation data
val_datagen = ImageDataGenerator(rescale = 1./255)
     


# In[52]:


X_test, y_test = read_and_process_image(test_imgs)
del test_imgs
gc.collect()


# In[53]:


plt.figure(figsize=(12, 8))
X_test = np.array(X_test)
y_test = np.array(y_test)
sns.countplot(y_val, saturation=1)
plt.title("Test image labels")
     


# In[54]:


# Create test and validation image generator
BATCH_SIZE = 64
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)
train_generator = train_datagen.flow(X_train, y_train, batch_size = BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size = BATCH_SIZE)


# In[82]:


# Train the model
#early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Entrenar el modelo
history1 = model1.fit(train_generator,
                              epochs=50,
                              validation_data=val_generator,
                              workers=8,
                              use_multiprocessing=False
                             )


# In[108]:


# Learning curves for training and validation
history_df1 = pd.DataFrame(history1.history)
history_df1
     


# In[109]:


plt.figure(figsize=(12, 8))
sns.lineplot(data=history_df1.loc[:, ["accuracy", "val_accuracy"]], palette=['b', 'r'], dashes=False)
sns.set_style("whitegrid")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
     


# In[110]:


X = np.array(X_test)
#sameeeee ends


# In[111]:


pred1 = model1.predict(X)
threshold = 0.5
predictions1 = np.where(pred1 > threshold, 1,0)


# In[112]:


test1 = pd.DataFrame(data = predictions1, columns = ["predictions"])
test1
test1["filename"] = [os.path.basename(i) for i in test_imgs]
test1["test_labels"] = y_test
test1 = test1[["filename", "test_labels", "predictions"]]
test1
     


# In[114]:


model_accuracy1 = accuracy_score(y_test, predictions1)
print("Model Accuracy: {:.2f}%".format(model_accuracy1 * 100))


# In[115]:


cl_report1 = classification_report(y_test, predictions1)
print(cl_report1)


# In[116]:


cn_matrix1= confusion_matrix(y_test, predictions1)
cn_matrix1


# In[117]:


f1, ax1 = plt.subplots(figsize = (8,6))
ax1 = sns.heatmap(cn_matrix1, annot=True,fmt="d")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")
ax1.set_title("Confusion Matrix")


# #Resnet Model Implementation

# In[58]:


# Calling pre-trained resnet model
base_model2 = ResNet50(include_top=False,weights='imagenet',input_shape=(150,150,3))


# In[59]:


# Let's see how many layers are in the resnet model
print("Number of layers in the base model: ", len(base_model2.layers))


# In[60]:


# Freeze the layers in pre-trained model, we don't need to train again
for layer in base_model2.layers:
   layer.trainable = False


# In[61]:


# Create our classifier model, connect pre-trained model Resnet to our model
model2 = keras.models.Sequential()
model2.add(base_model2)
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation = 'relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1, activation = 'sigmoid'))


# In[62]:


# Create summary of our model
model2.summary()


# In[63]:


model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[64]:


# Configure data augumentation and scaling of images to prevent overfitting since we have a small train data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 40,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

# Only rescaling for validation data
val_datagen = ImageDataGenerator(rescale = 1./255)
     


# In[77]:


X_test, y_test = read_and_process_image(test_imgs)
del test_imgs
gc.collect()


# In[78]:


plt.figure(figsize=(12, 8))
X_test = np.array(X_test)
y_test = np.array(y_test)
sns.countplot(y_val, saturation=1)
plt.title("Test image labels")
     


# In[79]:


# Create test and validation image generator
BATCH_SIZE = 64
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)
train_generator = train_datagen.flow(X_train, y_train, batch_size = BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size = BATCH_SIZE)


# In[83]:


# Train the model
#early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Entrenar el modelo
history2 = model2.fit(train_generator,
                              epochs=50,
                              validation_data=val_generator,
                              #callbacks=[early_stopping],
                              workers=8,
                              use_multiprocessing=False
                             )


# In[139]:


# Learning curves for training and validation
history_df2 = pd.DataFrame(history2.history)
history_df2
     


# In[140]:


plt.figure(figsize=(12, 8))
sns.lineplot(data=history_df2.loc[:, ["loss", "val_loss"]], palette=['b', 'r'], dashes=False)
sns.set_style("whitegrid")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")


# In[141]:


X = np.array(X_test)


# In[142]:


pred2 = model2.predict(X)
threshold = 0.5
predictions2 = np.where(pred2 > threshold, 1,0)


# In[143]:


test2 = pd.DataFrame(data = predictions2, columns = ["predictions"])
test2
test2["filename"] = [os.path.basename(i) for i in test_imgs]
test2["test_labels"] = y_test
test2 = test2[["filename", "test_labels", "predictions"]]
test2
     


# In[144]:


model_accuracy2 = accuracy_score(y_test, predictions2)
print("Model Accuracy: {:.2f}%".format(model_accuracy2 * 100))


# In[145]:


cl_report2 = classification_report(y_test, predictions2)
print(cl_report2)


# In[125]:


cn_matrix2= confusion_matrix(y_test, predictions2)
cn_matrix2


# In[126]:


f2, ax2 = plt.subplots(figsize = (8,6))
ax2 = sns.heatmap(cn_matrix2, annot=True,fmt="d")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
ax2.set_title("Confusion Matrix")







