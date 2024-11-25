import cv2
import numpy as np
from scipy.spatial.distance import cdist


# === 1. 이미지 전처리 및 로드 ===
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return img


# === 2. SIFT 특징점 검출 및 매칭 ===
def detect_and_match_sift(image1, image2, ratio_test_threshold=0.75):
    sift = cv2.SIFT_create()

    # 특징점 및 디스크립터 검출
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 매칭 수행 (KNN 매칭 사용)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's Ratio Test 적용
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)

    matched_points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    matched_points2 = [keypoints2[m.trainIdx].pt for m in good_matches]

    return matched_points1, matched_points2, keypoints1, keypoints2, good_matches


# === 3. RANSAC을 통한 매칭 정제 ===
def refine_matches_ransac(points1, points2, threshold=5.0):
    if len(points1) < 4 or len(points2) < 4:
        return points1, points2  # 매칭이 적으면 정제 불가

    points1 = np.array(points1)
    points2 = np.array(points2)

    # RANSAC을 사용해 이상치 제거
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, threshold)
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]

    return inliers1.tolist(), inliers2.tolist()


# === 4. Ground Truth 로드 ===
def detect_ground_truth(image, max_corners=400, quality_level=0.01, min_distance=10):
    corners = cv2.goodFeaturesToTrack(image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    if corners is not None:
        return [(int(c[0][0]), int(c[0][1])) for c in corners]
    return []


# === 5. 정확도 계산 ===
def calculate_accuracy(ground_truth, detected_points, threshold=5):
    if len(detected_points) == 0:
        return 0.0

    distances = cdist(ground_truth, detected_points, metric='euclidean')
    matches = (distances.min(axis=1) <= threshold).sum()
    accuracy = matches / len(ground_truth) if len(ground_truth) > 0 else 0.0
    return accuracy


# === 6. 결과 시각화 ===
def visualize_results(image1, image2, ground_truth, matched_points1, matched_points2, keypoints1, keypoints2, good_matches):
    # Ground Truth 시각화
    img_ground_truth = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    for pt in ground_truth:
        cv2.circle(img_ground_truth, pt, 5, (0, 255, 0), -1)  # 초록색

    # 매칭 시각화
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 결과 출력
    cv2.imshow("Ground Truth", img_ground_truth)
    cv2.imshow("SIFT Matches with Lowe's Ratio Test and RANSAC", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# === 7. 실행 ===
if __name__ == "__main__":
    image_path1 = "path/building.jpg"  # 비교할 이미지 1
    image_path2 = "path/building.jpg"  # 비교할 이미지 2

    # 이미지 로드 및 전처리
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    # Ground Truth 검출
    ground_truth = detect_ground_truth(img1)

    # SIFT 검출 및 매칭
    matched_points1, matched_points2, keypoints1, keypoints2, good_matches = detect_and_match_sift(img1, img2, ratio_test_threshold=0.8)

    # RANSAC 정제
    refined_points1, refined_points2 = refine_matches_ransac(matched_points1, matched_points2)

    # 정확도 계산
    sift_accuracy = calculate_accuracy(ground_truth, refined_points1)
    print(f"SIFT Accuracy after Lowe's Ratio Test and RANSAC: {sift_accuracy * 100:.2f}%")

    # 결과 시각화
    visualize_results(img1, img2, ground_truth, refined_points1, refined_points2, keypoints1, keypoints2, good_matches)
