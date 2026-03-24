def get_frames(cap):

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        yield frame