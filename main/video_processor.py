import cv2
from mtcnn.mtcnn import MTCNN

def detect_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

def draw_faces(image, faces):
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

# Video input
video_path = r"C:\Users\mahes\Downloads\WhatsApp Video 2024-01-26 at 00.45.09_6c002a23.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    draw_faces(frame, faces)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
