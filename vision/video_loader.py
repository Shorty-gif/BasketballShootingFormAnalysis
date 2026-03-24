import cv2

def load_video(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video")

    return cap