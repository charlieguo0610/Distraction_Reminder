from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_eye.xml')
distract_model = load_model('./model/distraction_model.hdf5', compile=False)

# some constants
frame_width = 1200
border = 2
min_width = 240
min_height = 240
min_width_eye = 60
min_height_eye = 60
scale_factor = 1.1
min_neighbours = 5

cv2.namedWindow("Don't Distract!")
camera = cv2.VideoCapture(0)
# check camera
if camera.isOpened() == False:
    print('The camera is not open')

# start main loop
while True:
    ret, frame = camera.read()
    if ret: # or we do not get image from camera
        frame = imutils.resize(frame, width=frame_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,
                                              minSize=(min_width, min_height),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                grayForDetect = gray[y:y+h, x:x+w]
                colorForShow = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(grayForDetect, scaleFactor=scale_factor,
                                                    minNeighbors=min_neighbours,minSize=(min_width_eye, min_height_eye))
                probs = list()
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(colorForShow, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), border)
                    sample = colorForShow[ey+border:ey+eh-border, ex+border:ex+ew-border]
                    # adjust for CNN
                    sample = cv2.resize(sample, (64, 64))
                    sample = sample.astype('float')/255.0 # normalize
                    sample = img_to_array(sample)
                    sample= np.expand_dims(sample, axis=0) # add a dimension
                    probability = distract_model.predict(sample)
                    probs.append(probability[0])
                avg = np.mean(probs)
                #get result
                if avg <= 0.5:
                    label = 'distracted'
                else:
                    label = 'focused'
                cv2.rectangle(frame, (x, y+h-30), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x, y + h - 5), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 1)

        cv2.imshow("Don't Distract!", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # no frame, don't do stuff
    else:
        break

# close
camera.release()
cv2.destroyAllWindows()
