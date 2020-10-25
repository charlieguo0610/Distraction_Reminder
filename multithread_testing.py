import numpy as np
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import threading
import arcade

face_cascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_eye.xml')
distract_model = load_model('./model/distraction_model1.hdf5', compile=False)

# some constants
frame_width = 1200
border = 2
min_width = 240
min_height = 240
min_width_eye = 60
min_height_eye = 60
scale_factor = 1.1
min_neighbours = 5
filename = 'alert.mp3'
cv2.namedWindow("Don't Distract!")


# default window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "TESTING"

try:
    sound = arcade.sound.load_sound("alert.mp3")
except Exception as e:
    print("Error loading sound.", e)


class MyGame(arcade.Window):
    """ Create a main application class (Inherit from arcade.Window class)
    """

    def __init__(self, width: int, height: int, title: str):
        """ Initialize the game window

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        super().__init__(width, height, title)
        self.distraction = False

    def draw_game(self) -> None:
        """ Draw everything in the game
        """
        arcade.draw_text('TESTING', 100, 100, color=arcade.color.GOLD, font_size=12)


    def on_draw(self) -> None:
        """ Render the screen.
        """
        arcade.start_render()
        self.draw_game()

    def setup(self) -> None:
        """ Initialize game interface. Default schedule is 60 fps.

        Args:
            None
        Returns:
            None
        """
        arcade.schedule(self.on_update, 1 / 60)
        self.frame_count = 0


    def update(self, delta_time: float) -> None:
        """ All the logic to move, and the game logic goes here

        Args:
            delta_time: time interval since the last time the function was called

        Returns:
            None
        """

        # Face Mode
        if self.frame_count % 30 == 0:

            if Communication.get_x() != '':
                self.distraction = True if Communication.get_x() == '1' else False
            else:
                print("face not detected")
            if self.distraction:
                arcade.play_sound(sound)

        # update the frame_count
        self.frame_count += 1


class Communication:
    """
    This is a Communication class that use for save data

    """

    def __init__(self, x: int, y: int):
        """ Initialize Communication class
        Args:
            x_value: x value that read out from Vision
            y_value: y value that read out from Vision
        Returns:
            None
        """
        self.x_value = x
        self.y_value = y

    def write_x(self) -> None:
        """ write x data in to database
        Args:
           x_value: x value that read out from Vision
           y_value: y value that read out from Vision
        Returns:
           None
        """
        with open("x.txt", "w") as f:
            f.write(str(self.x_value))
        with open("xb.txt", "w") as f:
            f.write(str(self.x_value))

    def write_y(self) -> None:
        """ write y data in to database
        Args:
            x_value: x value that read out from Vision
            y_value: y value that read out from Vision
        Returns:
            None
        """
        with open("y.txt", "w") as f:
            f.write(self.y_value)

        with open("yb.txt", "w") as f:
            f.write(self.y_value)

    @staticmethod
    def get_x() -> str:
        """ read x data from database
        Args:
           x_value: x value that read out from Vision
           y_value: y value that read out from Vision
        Returns:
           contents: x value in the database
        """
        try:
            with open("x.txt", 'r') as f:
                contents = f.read()
                return contents

        except:
            with open("xb.txt", 'r') as f:
                contents = f.read()
                return contents

    @staticmethod
    def get_y() -> str:
        """ read y data from database
        Args:
            x_value: x value that read out from Vision
            y_value: y value that read out from Vision
        Returns:
            contents: y value in the database
        """
        try:
            with open("y.txt", 'r') as f:
                y_contents = f.read()
                return y_contents
        except:
            with open("yb.txt", 'r') as f:
                y_contents = f.read()
                return y_contents


class Vision(Communication):

    def __init__(self, x: int, y: int):
        """
        Initialize Communication class
        Args:
            x_value: x value that read out from Vision
            y_value: y value that read out from Vision
        Return:
            None
        """
        super().__init__(x, y)

    @staticmethod
    def Face_Detect():
        """
        This is a vision function that use for
        face detection
        it trac your face and run CNN
        Args:
        cap: camra
        x = the x value your face on screen
        y = the y value your face on screen

        Return:
        None
        """
        cap = cv2.VideoCapture(0)  # 捕获摄像头图像

        # 判断视频是否打开

        if cap.isOpened():
            print('Open')
        else:
            print('camra is not opened')
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            if ret:  # or we do not get image from camera
                frame = imutils.resize(frame, width=frame_width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                                      minNeighbors=min_neighbours,
                                                      minSize=(min_width, min_height),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        grayForDetect = gray[y:y + h, x:x + w]
                        colorForShow = frame[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(grayForDetect, scaleFactor=scale_factor,
                                                            minNeighbors=min_neighbours,
                                                            minSize=(min_width_eye, min_height_eye))
                        probs = list()
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(colorForShow, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                                          border)
                            sample = colorForShow[ey + border:ey + eh - border,
                                     ex + border:ex + ew - border]
                            # adjust for CNN
                            sample = cv2.resize(sample, (64, 64))
                            sample = sample.astype('float') / 255.0  # normalize
                            sample = img_to_array(sample)
                            sample = np.expand_dims(sample, axis=0)  # add a dimension
                            probability = distract_model.predict(sample)
                            probs.append(probability[0])
                        avg = np.mean(probs)
                        # get result
                        if avg <= 0.5:
                            label = 'distracted'
                            distracted = 1
                            now = Communication(distracted, 0)
                            now.write_x()
                            with open('xb.txt', 'w') as f:
                                f.write(str(distracted))
                        else:
                            label = 'focused'
                            distracted = 0
                            now = Communication(distracted, 0)
                            now.write_x()
                            with open('xb.txt', 'w') as f:
                                f.write(str(distracted))
                        cv2.rectangle(frame, (x, y + h - 30), (x + w, y + h), (0, 255, 0),
                                      cv2.FILLED)
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


class MyThread(threading.Thread):
    """
    This is a class for Multithreading
    """

    def run(self):
        """This is a thread for Vision it start runing from there """
        print("Vision start")
        Vision.Face_Detect()


def main():
    vision = MyThread()
    # Start the threading
    vision.start()
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    SCREEN_TITLE = "Distraction Detection"
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()
    # Start the threading
    vision.join()
    print("End Main threading")


if __name__ == '__main__':
    main()