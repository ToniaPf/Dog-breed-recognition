import base64
import io

from flask import Flask, render_template, request, redirect
import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from detect import detect_in_image
from crop import crop

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

detector = load_model(os.path.join(basedir, 'detector.h5'))
predictor = load_model(os.path.join(basedir, 'inception_224_v2_ft.h5'))

dog_breeds = {
    0: 'Shiba dog', 1: 'French bulldog', 2: 'Siberian husky', 3: 'Malamute', 4: 'Pomeranian', 5: 'Airedale',
    6: 'Miniature poodle', 7: 'Affenpinscher', 8: 'Schipperke', 9: 'Australian terrier', 10: 'Welsh springer spaniel',
    11: 'Curly coated retriever', 12: 'Staffordshire bullterrier', 13: 'Norwich terrier', 14: 'Tibetan terrier',
    15: 'English setter', 16: 'Norfolk terrier', 17: 'Pembroke', 18: 'Tibetan mastiff', 19: 'Border terrier',
    20: 'Great dane', 21: 'Scotch terrier', 22: 'Flat coated retriever', 23: 'Saluki', 24: 'Irish setter',
    25: 'Blenheim spaniel', 26: 'Irish terrier', 27: 'Bloodhound', 28: 'Redbone', 29: 'West highland white terrier',
    30: 'Brabancon griffo', 31: 'Dhole', 32: 'Kelpie', 33: 'Doberman', 34: 'Ibizan hound', 35: 'Vizsla', 36: 'Cairn',
    37: 'German shepherd', 38: 'African hunting dog', 39: 'Dandie dinmont', 40: 'Sealyham terrier',
    41: 'German short haired pointer', 42: 'Bernese mountain dog', 43: 'Saint bernard', 44: 'Leonberg',
    45: 'Bedlington terrier', 46: 'Newfoundland', 47: 'Lhasa', 48: 'Chesapeake bay retriever', 49: 'Lakeland terrier',
    50: 'Walker hound', 51: 'American staffordshire terrier', 52: 'Otterhound', 53: 'Sussex spaniel',
    54: 'Norwegian elkhound', 55: 'Bluetick', 56: 'Dingo', 57: 'Irish water spaniel', 58: 'Samoyed',
    59: 'Fila braziliero', 60: 'Standard schnauzer', 61: 'Mexican hairless', 62: 'Entlebucher',
    63: 'Afghan hound', 64: 'Kuvasz', 65: 'English foxhound', 66: 'Keeshond', 67: 'Irish wolfhound',
    68: 'Scottish deerhound', 69: 'Rottweiler', 70: 'Black and tan coonhound', 71: 'Great pyrenees',
    72: 'Boxer', 73: 'Wire haired fox terrier', 74: 'Borzoi', 75: 'Groenendael', 76: 'Collie', 77: 'Gordon setter',
    78: 'Kerry blue terrier', 79: 'Briard', 80: 'Rhodesian ridgeback', 81: 'Boston bull', 82: 'Bull mastiff',
    83: 'Silky terrier', 84: 'Brittany spaniel', 85: 'Eskimo dog', 86: 'Giant schnauzer', 87: 'Malinois',
    88: 'Bouvier des flandres', 89: 'Whippet', 90: 'Appenzeller', 91: 'Chinese crested dog', 92: 'Miniature schnauzer',
    93: 'Soft coated wheaten terrier', 94: 'Weimaraner', 95: 'Clumber', 96: 'Greater swiss mountain dog',
    97: 'Toy terrier', 98: 'Italian greyhound', 99: 'Basset', 100: 'Basenji', 101: 'Australian shepherd',
    102: 'Maltese dog', 103: 'Japanese spaniel', 104: 'Cane carso', 105: 'Japanese spitzes',
    106: 'Old english sheepdog', 107: 'Black sable', 108: 'Border collie', 109: 'Shetland sheepdog',
    110: 'English springer', 111: 'Beagle', 112: 'Cocker spaniel', 113: 'Cardigan', 114: 'Toy poodle',
    115: 'Bichon frise', 116: 'Standard poodle', 117: 'Komondor', 118: 'Chow', 119: 'Chinese rural dog',
    120: 'Yorkshire terrier', 121: 'Labrador retriever', 122: 'Shih tzu', 123: 'Chihuahua', 124: 'Pekinese',
    125: 'Golden retriever', 126: 'Miniature pinscher', 127: 'Teddy', 128: 'Pug', 129: 'Papillon'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    f = request.files['files[]']
    if not f.filename:
        f = request.files.getlist('files[]')[1]
    file_path = os.path.join(basedir, 'images', f.filename)
    f.save(file_path)
    v_boxes, v_labels, v_scores = detect_in_image(detector, file_path, 'dog')
    results = []
    for v_box in v_boxes:
        cropped_image = crop(file_path, v_box).resize((224, 224))
        breed_probabilities = predictor.predict(np.array([img_to_array(cropped_image) / 255]))[0]
        top_three = np.argpartition(breed_probabilities, -3)[-3:]
        breeds = [dog_breeds[i] for i in top_three][::-1]
        scores = [float(breed_probabilities[i]) for i in top_three][::-1]
        scores_sorted, breeds_sorted = list(zip(*sorted(zip(scores, breeds), reverse=True)))
        data = io.BytesIO()
        cropped_image.save(data, "JPEG")
        i = base64.b64encode(data.getvalue()).decode('utf-8')
        results.append({'breeds': breeds_sorted, 'scores': scores_sorted, 'image': i})
    return {"success": True, 'results': results}, 200


