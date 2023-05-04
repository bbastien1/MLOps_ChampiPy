import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

# Import Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential

#  MLflow
import mlflow

# experiment_id = mlflow.create_experiment("VGG16+2")
experiment_name = "VGG16+2"
current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id=current_experiment['experiment_id']
mlflow.start_run(experiment_id =experiment_id)

mlflow.tensorflow.autolog()

tf.get_logger().setLevel('ERROR')

batch_size = 32
img_height = 160
img_width = 160
IMG_SIZE = (img_width, img_height)


root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root_dir, 'predict', 'images_random')

# Train Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
nb_classes = len(class_names)
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical",
        input_shape=(img_height,img_width,3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1)])

preprocess_input = tf.keras.applications.vgg16.preprocess_input
base_model = VGG16(weights='imagenet', input_shape=(img_height, img_width, 3), include_top=False)

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# Entete de classement
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Couche de prédiction
prediction_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
vgg = VGG16(weights='imagenet', input_shape=(img_height, img_width, 3), include_top=False, input_tensor=x)
x = global_average_layer(vgg.output)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

print("Trainable variables before :", len(model.trainable_variables))

for layer in model.layers[0:-1]:
    layer.trainable = False

print("Trainable variables after :", len(model.trainable_variables))


base_learning_rate = 0.1

# CALLBACKS
early_stopping = EarlyStopping(monitor = 'val_loss',
                            mode = 'min',
                            min_delta = 0.1,
                            patience = 15,
                            verbose = 1)


reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                                        min_delta = 0.01,
                                        patience = 8,
                                        factor = 0.1, 
                                        cooldown = 4,
                                        verbose = 1)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

# model.summary(show_trainable=True)

initial_epochs = 100

history = model.fit(train_ds, 
                    epochs=initial_epochs,
                    verbose=1,
                    callbacks=[early_stopping, reduce_learning_rate],
                    validation_data=val_ds)


# Courbe d'apprentissage

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

fig.savefig("train_VGG16plus2.png")
plt.close(fig)
