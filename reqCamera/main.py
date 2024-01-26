import cv2
import numpy as np

video = cv2.VideoCapture(0)

def preProcessing(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    imgPre = cv2.Canny(imgPre, 90, 140)
    kernel = np.ones((4, 4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=2)
    imgPre = cv2.erode(imgPre, kernel, iterations=2)
    return imgPre

while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))

    imgThres = preProcessing(img)

    countors, h1 = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contagem de contornos com área maior que 2000
    count_up2000 = 0

    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 2000:
            count_up2000 += 1
            cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(img, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                        2)

    # Exibindo a quantidade de objetos contornados na tela
    cv2.putText(img, "Objects: " + str(count_up2000), (10, 470), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Original", img)
    cv2.imshow("IMG", imgThres)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
