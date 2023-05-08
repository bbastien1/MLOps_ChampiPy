# Import des modules nécessaires
import os.path
import tensorflow as tf
import sys
import yaml
import shutil
import mlflow
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
from io import BytesIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from database.database import Database


def download_images_for_dataset(path:str, nb_img:int = 1000):

    chpy_db = Database()
    
    last_images = chpy_db.get_last_images(nb_img)
    
    # Suppression des données précédentes
    shutil.rmtree(path)

    # Téléchargement et enregistrement des images
    for grid_out in last_images:
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

def load_model(model_name: str="VGG16", stage: str = "Production"):

    mlruns_fld = os.path.realpath(os.path.join(SCRIPT_DIR, 'mlruns', 'models', model_name))

    for path, subdirs, files in os.walk(mlruns_fld):
        for name in files:    
            fullname = os.path.join(path, name)
            
            with open(fullname, 'r') as file:
                infos = yaml.safe_load(file)
                try:
                    if infos['current_stage'] == stage:
                        model_fld = infos['source']

                except KeyError:
                    # YAML dans model ne contient pas 'current_stage'
                    pass

    model_fld_split = model_fld.rsplit('mlruns')
    model_fld_fin = model_fld_split[1]
    model_fld_deb = os.path.realpath(os.path.join(SCRIPT_DIR, 'mlruns'))
    
    model_fld_final = model_fld_deb + model_fld_fin
    model_fld_final = os.path.realpath(model_fld_final)
    model_fld_final_tf = os.path.realpath(os.path.join(model_fld_final, "data", "model"))
    print("Model folder:", model_fld_final_tf)
    #model = mlflow.pyfunc.load_model(model_fld_final)

    model = tf.keras.models.load_model(model_fld_final_tf)
    return model


def fine_tune_model(model, nb_couches:int = 2):
    
    print("Trainable variables before :", len(model.trainable_variables))
    model.trainable = True

    for layer in model.layers[0:-nb_couches]:
        layer.trainable = False
    print("Trainable variables after :", len(model.trainable_variables))

    return model


def compile_model(model, learning_rate=0.0001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
    
    return model

def train_model(model, epochs:int, train_ds, val_ds):
    os.chdir(SCRIPT_DIR)
    experiment_name = "mlops_project"
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    mlflow.start_run(experiment_id = experiment_id)

    mlflow.tensorflow.autolog()
    mlflow.log_param("train_ds", train_ds)
    mlflow.log_param("val_ds", val_ds)

    tf.get_logger().setLevel('ERROR')

    # CALLBACKS
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

    history = model.fit(train_ds, 
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stopping, reduce_learning_rate],
                    validation_data=val_ds)
    
    return history


def get_history_last_values(history):
    ret={'accuracy': history.history['accuracy'][-1],
         'val_accuracy': history.history['val_accuracy'][-1],
         'loss': history.history['loss'][-1],
         'val_loss': history.history['val_loss'][-1]
        }
    return ret


def history_plot(history):
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

    fig.savefig("fine_tune_VGG16.png")
    plt.close(fig)


if __name__ == "__main__":
    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(root_dir, 'predict', 'images_temp')

    download_images_for_dataset(images_dir)
    train_ds, val_ds, nb_classes = create_datasets(images_dir) 
    model = load_model()

    model = fine_tune_model(model, 6)
    model = compile_model(model)
    history = train_model(model, 50, train_ds, val_ds)

    history_plot(history)