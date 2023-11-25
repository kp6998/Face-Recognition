import cv2
import face_recognition
import os
import glob

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing (adjust as needed)
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.jpg"))  # assuming images are in JPG format

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            # Get the filename only from the initial file path.
            filename = os.path.splitext(os.path.basename(img_path))[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = [(top * (1/self.frame_resizing), right * (1/self.frame_resizing),
                           bottom * (1/self.frame_resizing), left * (1/self.frame_resizing)) for (top, right, bottom, left) in face_locations]
        return face_locations, face_names


# Create an instance of SimpleFacerec
sfr = SimpleFacerec()

# Load encoding images from the "known_faces" folder
sfr.load_encoding_images("images/")

# Load camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect known faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Display results on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (int(left) + 6, int(bottom) - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
