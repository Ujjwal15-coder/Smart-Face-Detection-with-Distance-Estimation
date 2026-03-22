# import cv2
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# webcam = cv2.VideoCapture(0)
# while True:
#     _,img = webcam.read()
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,1.5,4)
#     for(x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#     cv2.imshow("Face detection",img)
#     key = cv2.waitKey(10)
#     if key == 27:
#         break
# webcam.release()
# cv2.destroyAllWindows()    


import cv2
import time

# Load face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
webcam = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

# Approx focal length (tune for your camera)
FOCAL_LENGTH = 615  
KNOWN_WIDTH = 14.0  # average face width in cm

def estimate_distance(face_width_in_frame):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / face_width_in_frame

while True:
    ret, img = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        distance = estimate_distance(w)

        # Label based on distance
        if distance < 40:
            status = "Too Close"
            color = (0, 0, 255)
        elif 40 <= distance <= 80:
            status = "Perfect"
            color = (0, 255, 0)
        else:
            status = "Too Far"
            color = (255, 0, 0)

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Show distance + status
        cv2.putText(img, f"{int(distance)} cm", (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(img, status, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Smart Face Detection", img)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
