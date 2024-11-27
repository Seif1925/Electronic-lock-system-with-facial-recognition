import cv2  # Imports the OpenCV library for computer vision tasks.
import threading  # Imports the threading library for concurrent task execution.
from pathlib import Path  # Imports the Path class from pathlib for working with file paths.
import serial  # Imports the serial library for serial communication with Arduino.
import time  # Imports the time module for time-related functions.
import smtplib  # Imports the smtplib library for sending email alerts.
import dlib  # Imports the dlib library for face detection and recognition tasks.
import numpy as np  # Imports the NumPy library for numerical computations.

# ESP32-CAM stream URL
stream_url = 'http://192.168.137.237:81/stream'

# Load dlib models
detector = dlib.get_frontal_face_detector()  # Loads the face detection model.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Loads the facial landmark predictor.

face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  
# Loads the face recognition model.

# Load reference images and create face descriptors
data_folder = Path('path_to_your_dataset')  
# Specifies the folder containing reference face images.

face_descriptors = []  # Initializes the list to store face descriptors.
face_images = []  # Initializes the list to store face images.

def send_email_alert():  
    # Compose the email message
    msg = EmailMessage()
    msg.set_content("Unauthorized access detected.")  # Sets the email message content.
    msg['Subject'] = "Security Alert"  # Sets the email subject.
    msg['From'] = "yourmail@gmail.com"  # Sender's email address.
    msg['To'] = "mymail@gmail.com"  # Recipient's email address.
    
    # Connect to the SMTP server, send the email, and close the connection
    server = smtplib.SMTP('smtp.gmail.com', 587)  # Connects to Gmail's SMTP server.
    server.starttls()  # Starts TLS encryption.
    server.login("yourmail@gmail.com", "your mail's app-specific password")  # Logs into the sender's email account.
    server.send_message(msg)  # Sends the email message.
    server.quit()  # Closes the SMTP server connection.

# Loop through reference images in the data folder
for file_path in data_folder.glob('*.[jp][pn]g'):  
    img = dlib.load_rgb_image(str(file_path))  # Loads the image using dlib.
    faces = detector(img)  # Detects faces in the image.
    if len(faces) > 0:  # If faces are detected in the image.
        shape = predictor(img, faces[0])  # Predicts facial landmarks for the detected face.
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)  
        # Computes the face descriptor.
        
        face_descriptors.append(np.array(face_descriptor))  # Adds the face descriptor to the list.
        face_images.append(img)  # Adds the face image to the list.

# Initialize serial communication with Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  
# Configures serial communication with Arduino.

def detect_faces():  
    cap = cv2.VideoCapture(stream_url)  # Opens the video stream from ESP32-CAM.
    while cap.isOpened():  # Loops while the video stream is open.
        ret, frame = cap.read()  # Reads a frame from the video stream.
        if not ret:  # If the frame cannot be read.
            break  # Exits the loop.
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converts the frame from BGR to RGB.
        faces = detector(frame_rgb)  # Detects faces in the frame.
        for face in faces:  # Loops through the detected faces.
            shape = predictor(frame_rgb, face)  # Predicts facial landmarks for the face.
            face_descriptor = face_rec_model.compute_face_descriptor(frame_rgb, shape)  
            # Computes the face descriptor.
            
            distances = np.linalg.norm(face_descriptors - np.array(face_descriptor), axis=1)  
            # Calculates distances between face descriptors.
            min_distance = np.min(distances)  # Finds the minimum distance.
            index_of_min_distance = np.argmin(distances)  # Finds the index of the minimum distance.

            if min_distance < 0.4:  # If the minimum distance is below the threshold.
                label = f"Authorized: {min_distance:.2f}"  # Sets the label for authorized access.
                color = (0, 255, 0)  # Green color for authorized access.
                ser.write(b'1')  # Signal to open the door.
            else:  # If the minimum distance is above or equal to the threshold.
                label = "Denied"  # Sets the label for denied access.
                color = (0, 0, 255)  # Red color for denied access.
                send_email_alert()  # Sends an email alert for unauthorized access.
            
            x, y, w, h = face.left(), face.top(), face.width(), face.height()  
            # Gets the coordinates of the face bounding box.
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  
            # Draws a rectangle around the face.
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
            # Adds the label to the frame.
        
        cv2.imshow('ESP32-CAM Stream', frame)  # Displays the frame with face detections.
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Waits for 'q' key press to exit.
            break  # Exits the loop.

    cap.release()  # Releases the video capture object.
    cv2.destroyAllWindows()  # Closes all OpenCV windows.

# Create and start a new thread for face detection
thread = threading.Thread(target=detect_faces)  
thread.start()  # Starts the thread for concurrent execution.
