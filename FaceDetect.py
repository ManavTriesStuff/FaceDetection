import cv2

cas_classifier = cv2.CascadeClassifier('HAARCascadeFaceDetection/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# Capture frame-by-frame
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, 0)
    detections = cas_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(150,0,150),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

#release the capture
cap.release()
cv2.destroyAllWindows()