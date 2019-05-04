import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
from shutil import copyfile
import pickle
import random

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions_used = ["neutral", "happy", "anger"]#, "sadness", "surprise"]
image_size = 250  # Pixels * pixels in image


def main():
    #sort_cohn_kanade_dataset()
    #clean_cohn_kanade_dataset()
    #create_data() #from cleaned data folder inot pickle file
    clean_fer2013()

# Sorts images into folders by their labelled emotion.
# Creates the 'new_data' folder
def sort_cohn_kanade_dataset():

    subjects = glob.glob("labels\\*") #List of subjects facial expression labels from labels folder
    for subject in subjects:
        subject_number = subject[-4:] #the subjects number from their folder
        for subject_image_folders in glob.glob("%s\\*" %subject): #Iterate through all the subjects folders
            for file in glob.glob("%s\\*" %subject_image_folders): #Iterate through each label in the folders
                current_photo_name = file[12:-30] #Save the name of the image
                label_txt = open(file, 'r')
                emotion = int(float(label_txt.readline())) #Read in emotion label from the text file

                image_path_emotion = glob.glob("images\\%s\\%s\\*" %(subject_number, current_photo_name))[-1] #-1 because the last image in the folder of images is the actual emotion facial expression
                image_path_neutral = glob.glob("images\\%s\\%s\\*" %(subject_number, current_photo_name))[0] #The first image in the sequence is a neutral pose, hence '0'

                new_image_path_emotion = "new_data\\%s\\%s" %(emotions[emotion], image_path_emotion[18:]) #Path for new emotion image in new_data folder sorted by its labelled emotion
                new_image_path_neutral = "new_data\\neutral\\%s" %image_path_neutral[18:] #Path for neutral image

                copyfile(image_path_emotion, new_image_path_emotion)
                copyfile(image_path_neutral, new_image_path_neutral)


#Find a face, crop and convert the data to gray scale. Populates the 'cleaned_data' folder
def clean_cohn_kanade_dataset():
    faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for emotion in emotions_used:    #Iterate through each emotion
        images = glob.glob("new_data\\%s\\*" %emotion) #list of all image files in that emotion fodler
        image_num = 0
        for image in images:
            grayscale = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            face_coords = faceClassifier.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in face_coords:
                grayscale = grayscale[y:y + h, x:x + w] #crop the image
                try:
                    processed_image = cv2.resize(grayscale, (image_size, image_size)) #resize the image
                    cv2.imwrite("cleaned_data\\%s\\%s.jpg" %(emotion, image_num), processed_image) #write the image to a file
                    image_num += 1
                except:
                    pass
                    image_num += 1

#Serialize the images and labels into pickle files to be used in training the prediction model
def create_data():
    training_data = []
    training_labels = []
    data = []

    for emotion in emotions_used:
        image_files = glob.glob("cleaned_data\\%s\\*" %emotion)
        #random.shuffle(image_files)

        for file in image_files:    #Create the training data
            image = cv2.imread(file)
            data.append( [ (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), (emotions_used.index(emotion)) ])
            #training_labels.append(emotions_used.index(emotion)) #Labels must be as an integer for the prediction model, not be a string

    #Shuffle the data and then create the X and Y training and label data
    random.shuffle(data)
    for x, y in data:
        training_data.append(x)
        training_labels.append(y)

    print(training_labels)
    pickle_out = open("X.pickle", "wb")     #Serialize the training data
    pickle.dump(training_data, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")      #Serialize the training labels
    pickle.dump(training_labels, pickle_out)
    pickle_out.close()

    return


def clean_fer2013():

    #Open and read the csv file containing the data and labels
    file = open("fer2013.csv")
    data = file.readlines()
    data_array = np.array(data)
    file.close()

    training_data = []
    training_labels = []
    test_data = []
    test_labels = []

    #Split the data into training, labels and test data
    for line in data_array[1:]:
        (emotion, image, type) = line.split(",")
        image_pixels = image.split(" ")
        image_array = np.array(image_pixels, 'float32')

        if (emotion == '1' or emotion == '2'):  # If disgust or fear discard the data
            continue
        elif (emotion == '6'):  # if neutral
            emotion = '0'
        elif (emotion == '3'):  # if happy
            emotion = '1'
        elif (emotion == '4'):  # if sad
            continue
        elif (emotion == '5'):  # if surprised
            emotion = '3'
        elif (emotion == '0'):  # if angry discard
            emotion = '2'


        # if (type == 'Training' or type == 'PublicTest'):
        if 'Training' in type:
            training_data.append(image_array)
            training_labels.append(emotion)
        elif 'PublicTest' in type:
            test_data.append(image_array)
            test_labels.append(emotion)


    pickle_out = open("Xtrain.pickle", "wb")     #Serialize the training data
    pickle.dump(training_data, pickle_out)
    pickle_out.close()
    pickle_out = open("Ytrain.pickle", "wb")      #Serialize the training labels
    pickle.dump(training_labels, pickle_out)
    pickle_out.close()

    pickle_out = open("Xtest.pickle", "wb")     #Serialize the test data
    pickle.dump(test_data, pickle_out)
    pickle_out.close()
    pickle_out = open("Ytest.pickle", "wb")      #Serialize the test labels
    pickle.dump(test_labels, pickle_out)
    pickle_out.close()

main()