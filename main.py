import numpy as np
import cv2
import time
import yaml

# number of frames for calibration
nCalFrames = 12
nFrames = 0
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space

imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)
previousTime = 0
gray = 0

while(True):
    # Capture frame-by-frame
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        if time.time() - previousTime > 2:
            previousTime = time.time()
            imgpoints.append(corners2)
            objpoints.append(objp)
            img = cv2.bitwise_not(img)
            nFrames = nFrames + 1

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,7), corners,ret)

    cv2.putText(img, '{}/{}'.format(nFrames, nCalFrames), (20, 460), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'press \'q\' to exit...', (255, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Webcam Calibration',img)
    if nFrames == nCalFrames:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# It's very important to transform the matrix to list.

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()