# -*- coding: utf-8 -*-
import Main
import fileCounter as fc

from collections import deque
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from tqdm import tqdm
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, AveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import warnings
import os
import glob
import matplotlib
import numpy as np
import pickle
import cv2

data_X, data_Y, data_id, data_type = Main.init_cam()

# Adattamento per caricamento da telecamere
PATH_cam = []

# Inizializzazione percorso telecamere
for i in range(1, 27):
    PATH_cam.append('dataset/' + str(i))

# Ciclare sulla lista di directory
# for x in PATH_cam:
#   print(x)

i = 1

for CAM in PATH_cam:

    print("\nCaricamento della Telecamera ", i)

    PATH_violence = (CAM + '/violenza')
    PATH_nonviolence = (CAM + '/nonViolenza')

    os.makedirs(('./data/Violence/' + str(i)), exist_ok=True)

    for path in tqdm(glob.glob(PATH_violence + '/*')):
        fname = os.path.basename(path).split('.')[0]
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 5 == 0:
                cv2.imwrite("./data/Violence/" + str(i) + "/{}-{}.jpg".format(fname, str(count).zfill(4)),
                            image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1

    os.makedirs(('./data/NonViolence/' + str(i)), exist_ok=True)
    for path in tqdm(glob.glob(PATH_nonviolence + '/*')):
        fname = os.path.basename(path).split('.')[0]
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 5 == 0:
                cv2.imwrite("./data/NonViolence/" + str(i) + "/{}-{}.jpg".format(fname, str(count).zfill(4)),
                            image)  # save frame as JPEG file
                success, image = vidcap.read()
            success, image = vidcap.read()
            count += 1

    i = i + 1

# quit()

# PRIMO TEST (SALTO DELLA CREAZIONE DEL MODEL)

fileName = r"model/violence_model.h5"
fileObj = Path(fileName)
if not (fileObj.is_file()):

    print("\nCreazione del modello in corso...\n")

    # Creating the video classification model

    matplotlib.use("Agg")

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    args = {
        "dataset": "data",
        "model": "model/violence_model.h5",
        "label-bin": "model/lb.pickle",
        "epochs": 1,
        "plot": "plot.png"

    }

    # initialize the set of labels from the spots activity dataset we are
    # going to train our network on
    LABELS = set(["Violence", "NonViolence"])

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print('-' * 100)
    print("[INFO] loading images...")
    print('-' * 100)
    imagePaths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in tqdm(imagePaths[::]):
        # imagePath : file name ex) V_123
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]  # Violence / NonViolence

        # if the label of the current image is not part of of the labels
        # are interested in, then ignore the image
        if label not in LABELS:
            continue

        # load the image, convert it to RGB channel ordering, and resize
        # it to be a fixed 224x224 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

    # Training Data Augmentation

    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # initialize the validation/testing data augmentation object (which
    # we'll be adding mean subtraction to)
    valAug = ImageDataGenerator()

    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainAug.mean = mean
    valAug.mean = mean

    # Load InceptionV3 model

    # load the InceptionV3 network, ensuring the head FC layer sets are left
    # off
    baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    model.trainable = True

    # Compile the model

    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable)
    print('-' * 100)
    print("[INFO] compiling model...")
    print('-' * 100)
    opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print(model.summary())

    # Train the model

    # train the head of the network for a few epochs (all other layers
    # are frozen) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random
    print('-' * 100)
    print("[INFO] training head...")
    print('-' * 100)
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=32),
        steps_per_epoch=len(trainX) // 32,
        validation_data=valAug.flow(testX, testY),
        validation_steps=len(testX) // 32,
        epochs=args["epochs"])

    # Evaluate the network

    # evaluate the network
    print('-' * 100)
    print("[INFO] evaluating network...")
    print('-' * 100)
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy
    print('-' * 100)
    print("[INFO] plot the training loss and accuracy...")
    print('-' * 100)
    N = args["epochs"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

    # serialize the model to disk
    print('-' * 100)
    print("[INFO] serializing network...")
    print('-' * 100)
    model.save(args["model"])

    # serialize the label binarizer to disk
    f = open(args["label-bin"], "wb")
    f.write(pickle.dumps(lb))
    f.close()

# FINE DELLA FASE DI TRAINING
# Predict the video for violence/ non-violence

print("\nModello Presente...\n")
print("TRAINING NON NECESSARIO\n")

CAM = 1

while CAM <= 26:

  dir = os.getcwd()
  partial_path = "/dataset/" + str(CAM) + "/nonViolenza/"

  CURRENT_PATH = dir + partial_path

  print(CURRENT_PATH)

  n_video = 1
  video_contenuti = fc.fcount(CURRENT_PATH)

  while n_video <= video_contenuti:

   print("\n Esamino CAM" + str(CAM) + " cartella NV, video numero " + str(n_video) + "\n")

   input_path = dir + "/dataset/" + str(CAM) + "/nonViolenza/" + str(n_video) + ".avi"
   print("\n Caricamento di " + input_path + "\n")

   output_path = dir + "/output/" + str(CAM) + "/nonViolenza/" + str(n_video) + "-esaminato.avi"
   print("\n Scrittura di " + output_path + "\n")

   os.makedirs((dir + "/output/" + str(CAM) + "/nonViolenza/"), exist_ok=True)

   args = {

    "model": "model/violence_model.h5",
    "label-bin": "model/lb.pickle",
    "input": input_path,
    "output": output_path,
    "size": 64

   }

   # load the trained model and label binarizer from disk
   print("[INFO] loading model and label binarizer...")
   model = load_model(args["model"])
   lb = pickle.loads(open(args["label-bin"], "rb").read())

   # initialize the image mean for mean subtraction along with the
   # predictions queue
   mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
   Q = deque(maxlen=args["size"])

  # initialize the video stream, pointer to output video file, and
  # frame dimensions
   vpath = args["input"]
   if args["input"] == 'camera':
      vpath = 0
   vs = cv2.VideoCapture(vpath)
   writer = None
   (W, H) = (None, None)

   # loop over frames from the video file stream
   while True:
      # read the next frame from the file
      (grabbed, frame) = vs.read()

      # if the frame was not grabbed, then we have reached the end
      # of the stream
      if not grabbed:
          break

      # if the frame dimensions are empty, grab them
      if W is None or H is None:
          (H, W) = frame.shape[:2]

      # clone the output frame, then convert it from BGR to RGB
      # ordering, resize the frame to a fixed 224x224, and then
      # perform mean subtraction
      output = frame.copy()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame, (224, 224)).astype("float32")
      frame -= mean

      # make predictions on the frame and then update the predictions
      # queue
      preds = model.predict(np.expand_dims(frame, axis=0))[0]
      Q.append(preds)

      # perform prediction averaging over the current history of
      # previous predictions
      results = np.array(Q).mean(axis=0)
      i = np.argmax(results)
      label = lb.classes_[i]

      # draw the activity on the output frame
      # prob = model.predict_proba(np.expand_dims(frame, axis=0))[0] # to show probability of frame
      prob = results[i] * 100

      text_color = (0, 255, 0)  # default : green

      if prob > 55:  # Violence prob
          text_color = (0, 0, 255)  # red
          text = "!!!WARNING!!! State : {:8} ({:3.2f}%)".format(label, prob)
       # ATTENZIONE VIOLENZA RILEVATA

      else:
          label = 'Normal'
          text = "State : {:8} ({:3.2f}%)".format(label, prob)

      # STAMPA FRASE SU FRAME DEL VIDEO
      # text = "State : {:8} ({:3.2f}%)".format(label, prob)
      FONT = cv2.FONT_HERSHEY_SIMPLEX

      cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

      # plot graph over background image
      output = cv2.rectangle(output, (35, 80), (35 + int(prob) * 5, 80 + 20), text_color, -1)

      # check if the video writer is None
      if writer is None:
          # initialize our video writer
          fourcc = cv2.VideoWriter_fourcc(*"MJPG")
          writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

      # write the output frame to disk
      writer.write(output)

      # show the output image
      #cv2.imshow("preview", output)
      key = cv2.waitKey(1) & 0xFF

      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
          break
   # release the file pointersq
   print("[INFO] cleaning up...")
   writer.release()
   vs.release()

   n_video = n_video + 1

  CAM = CAM + 1