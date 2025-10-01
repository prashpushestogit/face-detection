import cv2 as cv

face_cascade= cv.CascadeClassifier('haar_face.xml')
eye_cascade= cv.CascadeClassifier('haar_eye.xml')
smile_cascade= cv.CascadeClassifier('haar_smile.xml')

cap = cv.VideoCapture(0)

while True:

    ret,frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(frame, "Face", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv.putText(roi_color, "Eye", (ex, ey-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv.putText(roi_color, "Smile", (sx, sy-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv.imshow("Face, Eye & Smile Detection", frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


