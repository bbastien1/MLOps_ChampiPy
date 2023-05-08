import sys
import gridfs
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import mlflow

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from io import BytesIO
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from database.database import Database


def download_images_for_dataset(nb_img:int = 1000, path:str = ""):
    
    x_most_recent = fs.find().sort("uploadDate", -1).limit(nb_img)
    
    # Suppression des données précédentes
    shutil.rmtree(path)

    # Téléchargement et enregistrement des images
    for grid_out in x_most_recent:
        stream = BytesIO(grid_out.read())
        image_tmp = Image.open(stream).convert("RGB")
        
        image_folder = os.path.join(path, grid_out.classname)
        image_fullname = os.path.join(image_folder, grid_out.filename)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        image_tmp.save(image_fullname)


def create_datasets(path:str = "", ratio:float = 0.2, img_height:int = 120, img_width:int = 120, batch_size:int = 32):
    
    # Train Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
                                path,
                                validation_split=ratio,
                                subset="training",
                                seed=123,
                                image_size=(img_height, img_width),
                                batch_size=batch_size)

    # Validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
                                path,
                                validation_split=ratio,
                                subset="validation",
                                seed=123,
                                image_size=(img_height, img_width),
                                batch_size=batch_size)


    class_names = train_ds.class_names
    nb_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(50).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, nb_classes


def create_model(train_ds, nb_classes:int):

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

    model = fine_tune_model(model, 24)
    
    return model


def fine_tune_model(model, fine_tune_at:int = 26):
    
    print("Trainable variables before :", len(model.trainable_variables))
    model.trainable = True

    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    print("Trainable variables after :", len(model.trainable_variables))

    return model


def get_callbacks():
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                mode = 'min',
                                min_delta = 0.001,
                                patience = 15,
                                verbose = 1)

    reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                                            min_delta = 0.05,
                                            patience = 3,
                                            factor = 0.5, 
                                            cooldown = 4,
                                            verbose = 1)
    return early_stopping, reduce_learning_rate


if __name__ == "__main__":
        
    experiment_name = "mlops_project"
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    mlflow.start_run(experiment_id = experiment_id)

    mlflow.tensorflow.autolog()

    tf.get_logger().setLevel('ERROR')

    batch_size = 32
    img_height = 120
    img_width = 120
    IMG_SIZE = (img_width, img_height)

    base_learning_rate = 0.001
    initial_epochs = 50

    chpy_db = Database().DATABASE
    fs = gridfs.GridFS(chpy_db, 'img_store')

    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(root_dir, 'predict', 'images_temp')
    
    print("Dowloading images from DB...")
    download_images_for_dataset(1000, images_dir)
    
    print("Creating the datasets...")
    train_ds, val_ds, nb_classes = create_datasets(images_dir)   

    mlflow.log_param("train_ds", train_ds)
    mlflow.log_param("val_ds", val_ds)

    print("Preparing the model...")
    
    # Model
    model = create_model(train_ds, nb_classes)

    # CALLBACKS
    early_stopping, reduce_learning_rate = get_callbacks()

    

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    # model.summary(show_trainable=True)

    history = model.fit(train_ds, 
                        epochs=initial_epochs,
                        verbose=1,
                        callbacks=[early_stopping, reduce_learning_rate],
                        validation_data=val_ds)

    model = fine_tune_model(model, 20)

    fine_tune_epochs = 30
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_ds,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
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
    #plt.show()

    fig.savefig("train_VGG16.png")
    plt.close(fig)