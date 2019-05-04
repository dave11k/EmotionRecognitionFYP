import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import time
import cv2
import os
import tensorflow as tf
import pickle
from statistics import mode


model = tf.keras.models.load_model('final_model.model')
image_size = 48
cascade_classifier= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
window_name = "emotion"
predict_count = 0
current_prediction = "Neutral"

def main():
    #test = cv2.cvtColor(cv2.imread('93.jpg'), cv2.COLOR_BGR2GRAY)
    #prediction_image = test.reshape(-1, image_size, image_size, 1)
    #print(model.predict([prediction_image]))

    video_window = cv2.VideoCapture(0)
    video_window.set(3,720)
    video_window.set(4,1200)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 200, 20)
    cv2.resizeWindow(window_name, 870, 600)
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        run_video_stream(video_window)

def run_video_stream(video_window):
    global current_prediction
    global predict_count

    success, frame = video_window.read()

    #Pre processing convert to grayscale and equalize histogram
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    histogram_image = histogram.apply(grayscale_image)

    #Facial detection
    face_coords = cascade_classifier.detectMultiScale(histogram_image, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)  # Co-ords of face
    if(len(face_coords) == 1):
#        cv2.rectangle(frame, (face_coords[0][0], face_coords[0][1]),
#                      (face_coords[0][0] + face_coords[0][2], face_coords[0][1] + face_coords[0][3]), (0, 255, 0), 1)

        if(predict_count > 0):
            # Draw the border around the user's face and overlay the emoji for their predicted emotion
            box_colour = get_box_colour(current_prediction)
            cv2.rectangle(frame, (face_coords[0][0], face_coords[0][1]),
                          (face_coords[0][0] + face_coords[0][2], face_coords[0][1] + face_coords[0][3]), box_colour, 1)
            frame = overlay_emoji(frame, current_prediction, face_coords[0][0], face_coords[0][1])
        else:
            # Pre process the image of the face and get a prediction
            processed_image = process_image(histogram_image, face_coords)  # Send the face to pre processing
            current_prediction = predict(processed_image)

            box_colour = get_box_colour(current_prediction)
            cv2.rectangle(frame, (face_coords[0][0], face_coords[0][1]),
                          (face_coords[0][0] + face_coords[0][2], face_coords[0][1] + face_coords[0][3]), box_colour, 1)
            frame = overlay_emoji(frame, current_prediction, face_coords[0][0], face_coords[0][1])
            predict_count = 10


    predict_count -= 1
    cv2.putText(frame, "Press 'q' to quit", (20, 580), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    #Display the frame in the video window
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
        # close program

def overlay_emoji(background, emotion, x, y):
    file = cv2.imread("%s.png" %emotion, -1)
    emoji = cv2.resize(file, (image_size, image_size))

    y2 = y - emoji.shape[0]
    x2 = x + emoji.shape[1]

    if y2 < 0 or x2 < 0:
        return background

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_emoji
    for i in range(0, 3):
        background[y2:y, x:x2, i] = (alpha_emoji * emoji[:, :, i] + alpha_frame * background[y2:y, x:x2, i])

    return background

def get_box_colour(prediction):
    res = []

    if prediction == "Happy":
        res = (0, 255, 0)
    elif prediction == "Sad":
        res = (0, 0, 255)
    elif prediction == "Neutral":
        res = (255, 255, 255)
    elif prediction == "Surprised":
        res = (255, 255, 0)

    return res

#Return the most common prediction from the faces collected in the current interval
def predict(processed_image):
    prediction_image = processed_image.reshape(-1, image_size, image_size, 1)
    prediction = model.predict(prediction_image)
    prediction_string = get_prediction_string(prediction)

    print(prediction_string)
    return prediction_string


#Return a string from a prediction object
def get_prediction_string(prediction):
    index = np.where(prediction == np.amax(prediction))
    res = ""

    if index[1] == 0:
        res = "Neutral"
    elif index[1] == 1:
        res = "Happy"
    elif index[1] == 2:
        res = "Angry"
    elif index[1] == 3:
        res = "Surprised"

    return res

def delay_sad(video_window):
    ret, frame = video_window.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coords = cascade_classifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in face_coords:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #cv2.putText(frame, "Sad", (x - 1, y - 1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

        new_frame = overlay_emoji(frame, "Sad", x, y)
        cv2.imshow(window_name, new_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.imshow(window_name, frame)


#Returns the image frame cropped to just the face
def process_image(image, face):

    for (x, y, w, h) in face:
        new_y = int(y)
        new_h = int(h * 1.13)

        new_x = int(x + (w * 0.028))
        new_w = int(w * 0.95)
        processed_image = image[new_y:new_y + new_h, new_x:new_x + new_w]

    processed_image = cv2.resize(processed_image, (image_size, image_size))
    return processed_image


main()

