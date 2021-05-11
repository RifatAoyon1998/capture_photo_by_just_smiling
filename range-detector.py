# import cv2
# import numpy as np
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# def face_extractor(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     face=face_classifier.detectMultiScale(gray,1.3,5)
#
#     if face is():
#         return None
#
#     for(x,y,w,h) in face:
#         cropped_face = img[y:y+h, x:x+w]
#
#     return cropped_face
#
# cap= cv2.VideoCapture(0)
# count=0
#
# while True:
#     ret, frame = cap.read()
#     if face_extractor(frame) is not None:
#         count = count+1
#         fac = cv2.resize(face_extractor(frame),(200,200))
#         fac = cv2.cvtColor(fac,cv2.COLOR_BGR2GRAY)
#
#         file_name_path = 'Detect'+str(count)+'.jpg'
#         cv2.imwrite(file_name_path,fac)
#         cv2.putText(fac, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#         cv2.imshow('Face Cropper',fac)
#
#     else:
#         print("Face not found")
#         pass
#
#     if cv2.waitKey(1)==13 or count==20:
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()
# print('sample collected')
#
#
#
import cv2
import math
import argparse
#AGE find

# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
#
#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             faceBoxes.append([x1, y1, x2, y2])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
#     return frameOpencvDnn, faceBoxes
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--image')
#
# args = parser.parse_args()
#
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"
#
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# # ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(23-30)', '(35-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']
#
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)
#
# video = cv2.VideoCapture(args.image if args.image else 0)
# padding = 20
# while cv2.waitKey(1) < 0:
#     hasFrame, frame = video.read()
#     if not hasFrame:
#         cv2.waitKey()
#         break
#
#     resultImg, faceBoxes = highlightFace(faceNet, frame)
#     if not faceBoxes:
#         print("No face detected")
#
#     for faceBox in faceBoxes:
#         face = frame[max(0, faceBox[1] - padding):
#                      min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
#                                                                     :min(faceBox[2] + padding, frame.shape[1] - 1)]
#
#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderList[genderPreds[0].argmax()]
#         print(f'Gender: {gender}')
#
#         ageNet.setInput(blob)
#         agePreds = ageNet.forward()
#         age = ageList[agePreds[0].argmax()]
#         print(f'Age: {age[1:-1]} years')
#
#         # cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                     # (0, 255, 255), 2, cv2.LINE_AA)
#         cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                     (179,0,123), 2, cv2.LINE_AA)
#         cv2.imshow("Detecting age and gender", resultImg)
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