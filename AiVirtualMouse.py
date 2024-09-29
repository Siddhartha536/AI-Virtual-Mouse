import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
###########################
cap = cv.VideoCapture(0)

prvlocX, prevlocY = 0, 0
currlocX, currlocY = 0, 0

cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1 Find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2.Get the tip of the middle finger
    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5.Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen the value
            currlocX = prvlocX + (x3 - prvlocX) / smoothening
            currlocY = prevlocY + (y3 - prevlocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move((wScr - currlocX), currlocY)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            prvlocX, prevlocY = currlocX, currlocY

        # 8. Both Index and Middle fingers are up:Clicking None
        if fingers[1] == 1 and fingers[2] == 1:
            # 9.find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10.clicking mouse if distance is short
            if length < 20:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (20, 80), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    # 12. Display
    cv.imshow("Image", img)

    if cv.waitKey(20) == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
