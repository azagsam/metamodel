from tensorflow import keras
from pprint import pprint

model = keras.models.load_model('/home/azagar/myfiles/metamodel/model/metamodel-master/model.h5')
model.summary()