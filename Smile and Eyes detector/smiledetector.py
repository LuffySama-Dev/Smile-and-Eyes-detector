import cv2

# Ccollect the video
video = cv2.VideoCapture(0)

# Trained dataset
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

while True:
    
    # Catch the frame from the video
    (successful_frame_read, frame) = video.read()

    # Break if no frame 
    if not successful_frame_read:
        break

    # Converting to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting the faces
    face = face_detector.detectMultiScale(grayscale_frame)

    # Drawing the rectangle around face and smile
    for (x, y, w, h) in face:
        
        # Rectangle around face
        cv2.rectangle(frame, (x,y), (x+w , y+h), (0, 0, 255), 2)

        # Get the sub frame (Using numpy N-dimensional array slicing)
        the_face = frame[y:y+h , x:x+w]           

        # Change to grayscale        
        the_face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Create classifier
        smile = smile_detector.detectMultiScale(the_face_gray, scaleFactor=1.7, minNeighbors=20)
        eyes = eye_detector.detectMultiScale(the_face_gray, scaleFactor= 1.1, minNeighbors=10)

        # Find all the smile in the face
        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(the_face, (x_, y_), ( x_ + w_ , y_ + h_), (0, 255, 0), 2)

        # Label this face as smiling
        if len(smile) >0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (0, 0, 255))

        # Find eyes in the face
        for (x_, y_, w_, h_) in eyes:
            cv2.rectangle(the_face, (x_, y_), ( x_ + w_ , y_ + h_), (255, 255, 0), 2)

        

        

    # Show the captured frame 
    cv2.imshow('Simle Detector', frame)

    # Wait key
    key = cv2.waitKey(1)

    # Stopping functionality
    if key==81 or key==113:
        break

# Releasing the resources
video.release()
cv2.destroyAllWindows()

print('Code Completed')