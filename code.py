
import tensorflow as tf
import tensorflow_hub as hub
print("tensor flow version",tf.__version__)
print("version hub",hub.__version__)

print("GPU","Availabe,Yessssss" if tf.config.list_physical_devices("GPU") else "not available")

import pandas as pd
labels_csv=pd.read_csv("/content/drive/My Drive/Colab Notebooks/weed detection/archive (1).zip (Unzipped Files)/train_set_labels.csv")
print(labels_csv.describe)
print(labels_csv.head())

labels_csv["Species"].value_counts().plot.bar(figsize=(10,20))

labels_csv.head()

file_names=["/content/drive/My Drive/Colab Notebooks/weed detection/archive (1).zip (Unzipped Files)/train_set_labels.csv"+fname for fname in labels_csv["Label"]]
file_names[:10]

import os
if len(os.listdir("/content/drive/My Drive/Colab Notebooks/weed detection/archive (1).zip (Unzipped Files)/deepweeds_images_256"))==len(file_names):
  print("yep!!!!!priceed")
else:
  print("nope")

labels=labels_csv["Species"]
labels

import numpy as np
labels=labels_csv["Species"]
labels=np.array(labels)
labels


len(labels)

if len(labels)==len(file_names):
  print("yes")
else:
  print("nope")

unique_species=np.unique(labels)
unique_species
len(unique_species)

print(labels[0])
labels[0]==unique_species

# if labels==unique_species:
#   for i in labels:
#     print(labels)
#     print("true")
# else:
#   print("false")
#   print()

boolean_labels=[label==unique_species for label in labels]
boolean_labels[:3]

print(boolean_labels[0].astype(int))

x=file_names
y=boolean_labels
NUM_IMAGES=3000 #@param{type:"slider",min:1000 ,max:10000,step:1000}
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x[:NUM_IMAGES],
                                              y[:NUM_IMAGES],
                                              test_size=0.2,
                                              random_state=42)
len(x_train),len(x_val),len(y_train),len(y_val)

print(file_names[42])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/content/drive/My Drive/Colab Notebooks/weed detection/archive (1).zip (Unzipped Files)/deepweeds_images_256/20160928-140314-0.jpg')
print(img)
img.shape

img.max(), img.min()

tf.constant(img)

IMG_SIZE= 224

def process_image(image_path,img_size=IMG_SIZE):
  image=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image,channels=3)
  image=tf.image.convert_image_dtype(image,tf.float32)
  image=tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image


IMG_SIZE

def get_image_label(image_path,label):
  image=process_image(image_path)
  return image,label 

BATCH_SIZE=32
def crete_data_batches(x,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  if test_data:
    print("create the test data batches")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch=data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  elif valid_data:
    print("creating validation data batches..")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                            tf.constant(y)))
    data_batch=data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  else:
    print("create traing data batch")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                            tf.constant(y)))
    data=data.shuffle(buffer_size=len(x))
    data_batch=data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  

train_data=crete_data_batches(x_train,y_train)
val_data=crete_data_batches(x_val,y_val,valid_data=True)

train_data.element_spec,val_data.element_spec

y[0]

train_data

IMG_SIZE

import matplotlib.pyplot as plt
def show_25_images(images,labels):
  plt.figure(figsize=(10,10))
  for i in range(25):
    ax=plt.subplot(5,5,i+1)
    plt.imshow(images[i])
    plt.title(unique_species[labels[i].argmax()])

tensor=tf.io.read_file('/content/drive/My Drive/Colab Notebooks/weed detection/archive (1).zip (Unzipped Files)/deepweeds_images_256/20160928-140314-0.jpg')

len(boolean_labels)

INPUT_SHAPE=[None,IMG_SIZE,IMG_SIZE,3]
OUTPUT_SHAPE=len(unique_species)
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


INPUT_SHAPE

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

  # Build the model
  model.build(INPUT_SHAPE)

  return model


model = create_model()
model.summary()

%load_ext tensorboard



import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
  # Create a log directory for storing TensorBoard logs
  logdir = os.path.join("/content/drive/My Drive/Colab Notebooks/weed detection/log",
                        # Make it so the logs get tracked whenever we run an experiment
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)


NUM_EPOCHS=100 #@param {type:"slider",min:10,max:100,step:10}

print("GPU","Availabe,Yessssss" if tf.config.list_physical_devices("GPU") else "not available")

def train_model():
  model=create_model()
  tensor_board=create_tensorboard_callback()
  model.fit(x=train_data,
             epochs=NUM_EPOCHS,
             validation_data=val_data,
             validation_freq=1,
             callbacks=[tensor_board,early_stopping])
  return model
  

model=train_model()

from google.colab import drive
drive.mount('/content/drive')

