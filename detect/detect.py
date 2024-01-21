import cv2
import numpy as np
from config import MODEL_PATH
from detect.utils import detect_lp
from os.path import splitext
from keras.models import model_from_json
from PIL import Image


## Method to load Keras model weight and structure files
def load_model(path):
    try:
        path = splitext(path)[0]
        with open("%s.json" % path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights("%s.h5" % path)
        print("Model Loaded successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net = load_model(MODEL_PATH)


def preprocess_image(file, resize=False):
    image_array = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(file, Dmax=608, Dmin=608):
    vehicle = preprocess_image(file)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    LpImg = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg


def detect(file):
    LpImg = get_plate(file)

    image_data = LpImg[0]
    image_data = np.clip(image_data, 0, 1)

    # Scale the values to the range [0, 255] for RGB representation
    image_data *= 255
    image_data = image_data.astype(np.uint8)
    image = Image.fromarray(image_data)

    return image
