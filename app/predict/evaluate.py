import os.path
import tensorflow as tf
import mlflow
import mlflow.pyfunc
import tensorflow_datasets as tfds

from tensorflow import keras

def load_model(root_dir: str = ""):
    path = os.path.join(root_dir, "predict", "model")

    model = keras.models.load_model(path)
    return model


def get_accuracy():
    '''Return the accuracy evaluated of the trained model'''

    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    
    model = load_model(root_dir)
    eval_ds = get_eval_dataset(root_dir)

    loss0, accuracy0 = model.evaluate(eval_ds)
    
    return accuracy0


def get_eval_dataset(root_dir: str = ""):
    '''Return the evaluation dataset. Splited from get_accuracy to retrieve the classe names for predictions'''

    root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, 'train', 'images_eval')
    
    batch_size = 32
    img_height = 120
    img_width = 120

    # Evaluation Dataset
    eval_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    return eval_dataset


def get_classe_names(root_dir: str = ""):
    
    eval_ds = get_eval_dataset(root_dir)
    return eval_ds.class_names


def get_evaluation():
    model_name = "VGG16+2"
    stage = "Staging"

    model_uri=f"models:/{model_name}/{stage}"

    eval_data = get_eval_dataset()
    ds_data = tfds.as_dataframe(eval_data.take(10))
    # Log the baseline model to MLflow

    #model_uri = mlflow.get_artifact_uri("model")

    # Evaluate the logged model
    result = mlflow.evaluate(
        model_uri,
        ds_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    print(result)

if __name__ == "__main__":
    get_evaluation()