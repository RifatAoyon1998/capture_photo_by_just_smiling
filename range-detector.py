
import numpy as np
import cv2
# selfie by smile
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
font=cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)
img_counter = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for i in smile:
            if len(smile) > 1:

                # img_name = "opencv_frame_{}.png".format(img_counter)
                # cv2.imwrite(img_name, img)
                # print("{} written!".format(img_name))
                # img_counter += 1
               if (img_counter==0):
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, img)
                    print("{} written!".format(img_name))
                    cv2.putText(img, "Captured", (x, y - 10),font, 0.75, (255, 0, 255), 1,
                                cv2.LINE_AA)
                    img_counter += 1

        if (img_counter == 1):
            cv2.putText(img, "Captured", (x, y - 10), font, 0.75, (255, 0, 255), 1,
                        cv2.LINE_AA)

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
