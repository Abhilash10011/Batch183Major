import cv2
import argparse
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True, help='b.jpg')
args = vars(ap.parse_args())

# Load model from JSON file
json_file = open('C:\\Users\\abhilash talluri\\Downloads\\Facial-emotion-recognition-master-20240411T083201Z-001\\Facial-emotion-recognition-master\\top_models\\blocks_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('C:\\Users\\abhilash talluri\\Downloads\\Facial-emotion-recognition-master-20240411T083201Z-001\\Facial-emotion-recognition-master\\top_models\\top_models\\fer.h5')

classifier = cv2.CascadeClassifier('C:\\Users\\abhilash talluri\\Downloads\\Facial-emotion-recognition-master-20240411T083201Z-001\\Facial-emotion-recognition-master\\top_models\\haarcascade_frontalface_default.xml')

# Load input image
img = cv2.imread(args['image'])
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces_detected = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces_detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray_img[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0

    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions)

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    predicted_emotion = emotions[max_index]

    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# Display the result
cv2.imshow('Facial Emotion Recognition', img)
print("Displaying result...")

# Wait for key press and close window
cv2.waitKey(0)
cv2.destroyAllWindows()
