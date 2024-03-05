from flask import Flask , request
from flask_cors import CORS
from flask import jsonify
from flask import request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
model = load_model('model.h5')
model.make_predict_function()

classes =['affenpinscher',
 'afghan_hound',
 'african_hunting_dog',
 'airedale',
 'american_staffordshire_terrier',
 'appenzeller',
 'australian_terrier',
 'basenji',
 'basset',
 'beagle',
 'bedlington_terrier',
 'bernese_mountain_dog',
 'black-and-tan_coonhound',
 'blenheim_spaniel',
 'bloodhound',
 'bluetick',
 'border_collie',
 'border_terrier',
 'borzoi',
 'boston_bull',
 'bouvier_des_flandres',
 'boxer',
 'brabancon_griffon',
 'briard',
 'brittany_spaniel',
 'bull_mastiff',
 'cairn',
 'cardigan',
 'chesapeake_bay_retriever',
 'chihuahua',
 'chow',
 'clumber',
 'cocker_spaniel',
 'collie',
 'curly-coated_retriever',
 'dandie_dinmont',
 'dhole',
 'dingo',
 'doberman',
 'english_foxhound',
 'english_setter',
 'english_springer',
 'entlebucher',
 'eskimo_dog',
 'flat-coated_retriever',
 'french_bulldog',
 'german_shepherd',
 'german_short-haired_pointer',
 'giant_schnauzer',
 'golden_retriever',
 'gordon_setter',
 'great_dane',
 'great_pyrenees',
 'greater_swiss_mountain_dog',
 'groenendael',
 'ibizan_hound',
 'irish_setter',
 'irish_terrier',
 'irish_water_spaniel',
 'irish_wolfhound',
 'italian_greyhound',
 'japanese_spaniel',
 'keeshond',
 'kelpie',
 'kerry_blue_terrier',
 'komondor',
 'kuvasz',
 'labrador_retriever',
 'lakeland_terrier',
 'leonberg',
 'lhasa',
 'malamute',
 'malinois',
 'maltese_dog',
 'mexican_hairless',
 'miniature_pinscher',
 'miniature_poodle',
 'miniature_schnauzer',
 'newfoundland',
 'norfolk_terrier',
 'norwegian_elkhound',
 'norwich_terrier',
 'old_english_sheepdog',
 'otterhound',
 'papillon',
 'pekinese',
 'pembroke',
 'pomeranian',
 'pug',
 'redbone',
 'rhodesian_ridgeback',
 'rottweiler',
 'saint_bernard',
 'saluki',
 'samoyed',
 'schipperke',
 'scotch_terrier',
 'scottish_deerhound',
 'sealyham_terrier',
 'shetland_sheepdog',
 'shih-tzu',
 'siberian_husky',
 'silky_terrier',
 'soft-coated_wheaten_terrier',
 'staffordshire_bullterrier',
 'standard_poodle',
 'standard_schnauzer',
 'sussex_spaniel',
 'tibetan_mastiff',
 'tibetan_terrier',
 'toy_poodle',
 'toy_terrier',
 'vizsla',
 'walker_hound',
 'weimaraner',
 'welsh_springer_spaniel',
 'west_highland_white_terrier',
 'whippet',
 'wire-haired_fox_terrier',
 'yorkshire_terrier']

def preprocess_image(img):
    input_shape = (331, 331, 3)
    img = Image.open(io.BytesIO(img.read()))
    img = img.resize((input_shape[0], input_shape[1]))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img



@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = preprocess_image(request.files['image'])
        prediction = model.predict(img)
        print(f"Predicted label: {classes[np.argmax(prediction[0])]}")
        print(f"Probability of prediction): {round(np.max(prediction[0])) * 100} %")

        label = classes[np.argmax(prediction[0])]
        return jsonify({'pred': label})

if __name__ == '__main__':
    app.run()
