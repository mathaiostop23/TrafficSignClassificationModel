import cv2
import numpy as np
import tensorflow as tf
import time

# Load the saved model
model = tf.keras.models.load_model('TrafficSign.h5')

# Define the video capture object
cap = cv2.VideoCapture(0)

# Set the resolution of the video
cap.set(3, 1920)
cap.set(4, 1080)

# Define the class labels
class_labels = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']


# Define a function to preprocess the image
def preprocess_image(img):
    # Resize the image to 30x30
    img = cv2.resize(img, (30, 30))
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    img = img / 255.0
    # Reshape the image to a 4D tensor
    img = np.reshape(img, (1, 30, 30, 1))
    return img

def classify_traffic_sign(frame):
    if frame.ndim == 2:  # grayscale image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:  # single channel image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:  # already an RGB image
        rgb_frame = frame
    # Resize the frame to the input shape of the model
    img = cv2.resize(rgb_frame, (30, 30))
    # Normalize the pixel values
    img = img.astype(np.float32) / 255.0
    # Add batch dimension to the input
    img = np.expand_dims(img, axis=0)
    # Use the model to predict the class probabilities
    class_probs = model.predict(img)[0]
    # Get the index of the class with highest probability
    class_index = np.argmax(class_probs)
    # Get the label of the predicted class
    class_label = class_labels[class_index]
    # Get the probability of the predicted class
    class_prob = class_probs[class_index]
    # Return the label and probability of the predicted class
    return class_label, class_prob

# Start the video capture loop
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    # If the frame is valid
    if ret:
        # Get the predicted class label and probability
        class_label, class_prob = classify_traffic_sign(frame)
        # Draw the predicted class label and probability on the frame
        if class_prob > 0.9 :
            cv2.putText(frame, '{}: {:.2f}'.format(class_label, class_prob), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else :
            cv2.putText(frame, 'Traffic Sign Not Found', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          
        # Show the frame
        cv2.imshow('Video', frame)
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
