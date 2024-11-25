import cv2
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # 거리 계산 함수

# === 1. Ground Truth 데이터 불러오기 ===
def load_ground_truth(csv_path):
    try:
        # CSV 파일 읽기
        data = pd.read_csv(csv_path)
        # 좌표를 리스트로 변환 [(y1, x1), (y2, x2), ...]
        ground_truth = [(int(row['y']), int(row['x'])) for _, row in data.iterrows()]
        print("Ground Truth Loaded:", ground_truth)
        return ground_truth
    except Exception as e:
        print(f"Error loading Ground Truth: {e}")
        sys.exit()

# Ground Truth 데이터 경로
ground_truth_path = 'C:/Users/kim/PycharmProjects/pythonProject1/ground_truth.csv'
ground_truth = load_ground_truth(ground_truth_path)

# === 2. 이미지 로드 ===
src = cv2.imread('C:/Users/kim/Downloads/building.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# === 3. FastFeatureDetector 검출 ===
fast = cv2.FastFeatureDetector_create(60)  # 임계값 60 설정
keypoints = fast.detect(src)
dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)  # 컬러 이미지로 변환 (시각화용)

fast_features_points = []  # FAST 특징점 좌표 저장
for kp in keypoints:
    pt = (int(kp.pt[1]), int(kp.pt[0]))  # (y, x)
    fast_features_points.append(pt)
    cv2.circle(dst2, (pt[1], pt[0]), 5, (0, 0, 255), 2)

# === 4. 정확도 계산 함수 ===
def calculate_accuracy(ground_truth, detected_points, threshold=5):
    # 두 좌표 간 거리 계산
    distances = cdist(ground_truth, detected_points, metric='euclidean')
    # 매칭된 점의 수 계산 (거리 threshold 이내)
    matches = (distances.min(axis=1) <= threshold).sum()
    # 정확도 계산
    accuracy = matches / len(ground_truth)
    return accuracy

# === 5. 정확도 계산 ===
fast_features_accuracy = calculate_accuracy(ground_truth, fast_features_points, threshold=5)
print(f"FastFeatureDetector Accuracy: {fast_features_accuracy * 100:.2f}%")

# === 6. 결과 시각화 ===
cv2.imshow('Original Image', src)
cv2.imshow('FastFeatureDetector', dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()
