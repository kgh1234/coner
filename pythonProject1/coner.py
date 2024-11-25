import cv2
import sys
import numpy as np


src = cv2.imread('C:/Users/kim/Downloads/building.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load faile!')
    sys.exit()

# 좋은 특징점 검출 방법
corners = cv2.goodFeaturesToTrack(src, 400, 0.01, 10)

goodFeaturesToTrack = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

if corners is not None:
    for i in range(corners.shape[0]):  # 코너 갯수만큼 반복문
        pt = (int(corners[i, 0, 0]), int(corners[i, 0, 1]))  # x, y 좌표 받아오기
        cv2.circle(goodFeaturesToTrack, pt, 5, (0, 255, 0), 2)  # 받아온 위치에 원

# Fast 코너 검출
fast = cv2.FastFeatureDetector_create(60)  # 임계값 60 지정
keypoints = fast.detect(src)  # Keypoint 객체를 리스트로 받음

FastFeatureDetector = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for kp in keypoints:
    pt = (int(kp.pt[0]), int(kp.pt[1]))  # kp안에 pt좌표가 있음
    cv2.circle(FastFeatureDetector, pt, 5, (0, 0, 255), 2)

cv2.imshow('src', src)
cv2.imshow('goodFeaturesToTrack',goodFeaturesToTrack)
cv2.imshow('FastFeatureDetector', FastFeatureDetector)
cv2.waitKey()

cv2.destroyAllWindows()