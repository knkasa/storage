import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNN with image generator.  https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# Download data  https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
os.chdir("C:/my_working_env/test")

#================================================================================================

# Get directory info.
for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)


# Set the seed
tf.random.set_seed(42)
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
valid_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
# Setup the train and test directories
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

# Import data from directories and turn it into batches.  Use train_datagen.flow() if images are numpy format.
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

# Get the images of the preprocessed data. 
images, labels = train_data.next()                              

# Create a CNN model (same as Tiny VGG - https://poloclub.github.io/cnn-explainer/)
# If validation loss goes up, likely overfitting.  Use maxPool2D, add more data (by rotating, etc).
model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10, 
                         kernel_size=3, # can also be (3, 3)
                         activation="relu", 
                         input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
# Batch size = Number of images to send to Loss func before updating weights.
# Steps_per_epoch = # of images divide by batch size (default) = updating weights once means one epoch.
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),  # Number of images to send to Loss func before updating weights.
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
                        
import pdb; pdb.set_trace()  

  
