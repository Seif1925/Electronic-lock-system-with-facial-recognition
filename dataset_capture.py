import cv2
import pathlib

# Define the path to save images
data_path = pathlib.Path('path_to_your_dataset')
data_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

def save_image(img, id, img_id):
    filename = data_path / f"user.{id}.{img_id}.jpg"
    cv2.imwrite(str(filename), img)

def draw_rectangle(frame, classifier, scaleFactor, minNeighbors, color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, width, height) in features:
        cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
        coords = [x, y, width, height]
    return coords, frame

def detect(frame, face_cascade, img_id):
    color = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
    coords, frame = draw_rectangle(frame, face_cascade, 1.1, 10, color["blue"])
    if len(coords) == 4:
        roi = frame[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        save_image(roi, user_id, img_id)
    return frame

# Load the Haar Cascade for face detection
cascade_path = pathlib.Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(str(cascade_path))

# Setup the video capture device, use the ESP-CAM URL with correct port and endpoint
camera = cv2.VideoCapture('http://192.168.137.128:81/stream')

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

img_id = 0

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        continue

    frame = detect(frame, face_cascade, img_id)
    cv2.imshow("Faces", frame)
    img_id += 1

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

