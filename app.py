from flask import Flask, request
from waitress import serve
import numpy as np
from tensorflow import keras
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_back():
    rgb = [255, 255, 255]
    
    row = []
    for i in range(192):
        row.append(rgb)
        
    img = []
    for j in range(192):
        img.append(row)
    
    return np.array(img)
    

model_loaded = keras.models.load_model('gr_mobilenet2_model')
img_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
    )
#back = cv2.imread('background.jpg')

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello friend!!!"

@app.route('/letter')
def recognize_gesture():
    xstr = request.args.getlist('x')
    ystr = request.args.getlist('y')

    xlist = list(map(float, xstr))
    ylist = list(map(float, ystr))

    hand_landmarks = landmark_pb2.NormalizedLandmarkList()
    for i in range(len(xlist)):
        hand_landmarks.landmark.add(x=xlist[i], y=ylist[i])

    margin = 10
    minX = min(hand_landmarks.landmark, key=lambda i: i.x).x
    maxX = max(hand_landmarks.landmark, key=lambda i: i.x).x
    minY = min(hand_landmarks.landmark, key=lambda i: i.y).y
    maxY = max(hand_landmarks.landmark, key=lambda i: i.y).y

    w = maxX - minX
    h = maxY - minY

    maxLength = max(w, h)

    for i in range(len(hand_landmarks.landmark)):
        hand_landmarks.landmark[i].x = ((hand_landmarks.landmark[i].x + maxLength - maxX) * (192 - 2 * margin) / maxLength + margin) / 192
        hand_landmarks.landmark[i].y = ((hand_landmarks.landmark[i].y - minY) * (192 - 2 * margin) / maxLength + margin) / 192

    res_img = cv2.imread('background192.jpg')

    mp_drawing.draw_landmarks(
            res_img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    #res_img = cv2.flip(res_img, 1)
    
    res_imgs = img_generator.flow(np.array([res_img.tolist()]))
    pred = model_loaded.predict(res_imgs)
    letter = np.argmax(pred,axis=1)

    return ' '.join(str(i) for i in pred[0])

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
