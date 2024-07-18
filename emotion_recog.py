from facial_emotion_recognition import EmotionRecognition #f_e_r lib is imported to ER module
import cv2
er=EmotionRecognition(device='cpu')#load lib
cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frame=er.recognise_emotion(frame,return_type='BGR')
    cv2.imshow("Emotion Recognization",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
