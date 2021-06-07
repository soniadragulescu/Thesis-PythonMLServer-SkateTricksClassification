import urllib.request
from utils.Firebase import db, push_service, topic
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
import cv2
import math
import os
from glob import glob
from scipy import stats as s
import time

base_model = VGG16(weights='imagenet', include_top=False)

model = Sequential()

model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.load_weights("weights_skate.hdf5")

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

download_path = 'downloads/test_'
index = 0
classes = pd.get_dummies(['Fail', 'Slide', 'Ollie'])

server_trials = 1
while True:
    time.sleep(10)
    print("Starting server trial number: "+str(server_trials))
    server_trials += 1
    docs = db.collection('tricks').where("labeled", "==", False).get()
    if docs:
        print("There are "+str(len(docs))+" unlabeled docs.")
        for d in docs:
            url_link = d.to_dict()['url']
            videoFile = download_path + str(index) + ".mp4"
            urllib.request.urlretrieve(url_link, videoFile)
            index += 1
            key = d.id
            count = 0

            cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
            frameRate = cap.get(5)  # frame rate

            # removing all other files from the temp folder
            files = glob('datas/temp/*')
            for f in files:
                os.remove(f)

            print("Extracting the frames.")
            while cap.isOpened():
                frameId = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if not ret:
                    break
                if frameId % math.floor(frameRate) == 0:
                    # storing the frames of this particular video in temp folder
                    filename = 'datas/temp/' + "_frame" + str(count) + ".jpg"
                    count += 1
                    cv2.imwrite(filename, frame)
            cap.release()

            # reading all the frames from temp folder
            images = glob("datas/temp/*.jpg")

            prediction_images = []
            for i in range(len(images)):
                img = image.load_img(images[i], target_size=(224, 224, 3))
                img = image.img_to_array(img)
                img = img / 255
                prediction_images.append(img)

            print("Beginning the prediction.")
            # converting all the frames for a test video into numpy array
            prediction_images = np.array(prediction_images)
            # extracting features using pre-trained model
            prediction_images = base_model.predict(prediction_images)
            # converting features in one dimensional array
            prediction_images = prediction_images.reshape(prediction_images.shape[0], 7 * 7 * 512)
            # predicting tags for each array
            prediction = model.predict_classes(prediction_images)

            predict = classes.columns.values[s.mode(prediction)[0][0]]

            db.collection('tricks').document(key).update({"labeled": True, "tags": ["all", predict.lower()]})
            os.remove(videoFile)
            push_service.notify_topic_subscribers(topic_name=topic, message_body="A new video has been added to the "+predict.upper()+" category!")
            print("Prediction done! Class predicted: "+predict.upper()+". Frames removed, notification sended.")