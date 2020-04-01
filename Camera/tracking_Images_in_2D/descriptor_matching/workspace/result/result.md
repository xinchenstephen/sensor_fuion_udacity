# 1.Load the 'BRISK_small' dataset with cross-check first turned off, then on. Look at the visualized keypoint matches and at the number of matched pairs and describe your results.
non-cross-check : BF matching cross-check=0 (NN) with n=100 matches in 12.6026 ms
after-cross-check: BF matching cross-check=1 (NN) with n=56 matches in 1.49973 ms

# 2.dd the k-nearest-neighbor matching (using cv::knnMatch) with k=2 and implement the above-described descriptor distance ratio to filter out ambiguous matches with the threshold set to 0.8. Visualize the results, count the percentage of discarded matches (for both the 'BRISK_small' and the 'BRISK_large' dataset) and describe your observations.

for BRISK_small dataset:

BF matching cross-check=0 (KNN) with n=100 matches in 1.36263 ms
keypoints removed = 45

for BRISK_large  dataset:

BF matching cross-check=0 (KNN) with n=2896 matches in 96.3682 ms
keypoints removed = 1318

conclusion: using KNN can largely decrese the mismatching of the datapoint

# 3.Use both BF matching and FLANN matching on the 'BRISK_large' dataset and on the SIFT dataset and describe your observation

"BRISK_large" dataset:
BF:
BF matching cross-check=0 (KNN) with n=2896 matches in 96.3682 ms
keypoints removed = 1318

FLANN:
FLANN matching (KNN) with n=2896 matches in 56.1965 ms
keypoints removed = 1719

"SIFT" dataset:

BF:
BF matching cross-check=0 (KNN) with n=1890 matches in 67.9843 ms
keypoints removed = 860

FLANN:
FLANN matching (KNN) with n=1890 matches in 44.489 ms
keypoints removed = 842


conclusion: FLANN can largely increase the speed of the detection.


