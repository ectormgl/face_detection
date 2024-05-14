import cv2
#print(cv2.__version__)

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_cap= cv2.VideoCapture(0)

while True:
    ok, frame= video_cap.read()
    imagem_cinza= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes= detector.detectMultiScale(imagem_cinza)

    for x, y, w, h in deteccoes:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break


video_cap.release()
cv2.destroyAllWindows()