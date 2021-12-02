# Import kivy dependencies first
from kivy.app import App 

# Import kivy UX components
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
class CamApp(App):
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verfication_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        #Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verfication_label)

        # Load tensorflow model
        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist':L1Dist})


        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        # Flip horizontal and convert the image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create((frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image file and convert to 100x100px
    def preprocess(self, file_path):
        #Read in image file from path
        img = tf.io.read_file(file_path)
        #Load in image
        img = tf.io.decode_jpeg(img)
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1
        img = img/255.0
        
        # Return image
        return img

    # Verification function 
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.8
        verification_threshold = 0.8

        #Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data','input_image', 'input_image.jpg')
        verification_image_path = os.path.join('application_data', 'verification_images')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        input_image = self.preprocess(SAVE_PATH)
        
        # Build result array
        results = []
        # Prediction with each image in the verification_image folder
        for image in os.listdir(verification_image_path):
            validation_image = self.preprocess(os.path.join(verification_image_path, image))
            result = self.model.predict(list(np.expand_dims([input_image, validation_image], axis=1)))
            results.append(result)  

        # Detection Threshold: Metrics above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
    #     print("detection:", detection)
        
        # Varification Thershold: ratio of positive predictions / total positives samples
        verification = detection / len(os.listdir(verification_image_path))
        verified = verification > verification_threshold

        # Set verification text
        self.verfication_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        #Logger.info(results)
        
        return results, verified
 
if __name__ == '__main__':
    CamApp().run()