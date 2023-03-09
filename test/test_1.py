from tensorflow import keras

def load_index_to_label(path: str = ''):
        index_to_class_label = pd.read_csv('../final_500_targets.csv', sep = ';')
        return index_to_class_label

def load_model(path: str = '../model'):
    model = keras.models.load_model(path)
    return model

def predict(img):   
    prediction = model.predict(img)
    prediction = tf.nn.softmax(prediction[0])
    return prediction

