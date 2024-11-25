# corner


Project Overview
Problem: When comparing the FAST and Good Features to Track (Good) algorithms, FAST showed low accuracy due to its sensitivity to noise.

Solution: On the other hand, Good Features to Track demonstrated relatively higher accuracy, so it was chosen as the Ground Truth. Using SIFT (Scale-Invariant Feature Transform), we aimed to improve the accuracy of FAST.

Key Results
Initial Comparison: The initial accuracy of FAST was approximately 49%, which was significantly lower compared to Good Features to Track.
After Applying SIFT: By applying SIFT to reduce noise and improve matching quality, the accuracy of FAST increased to 82%.

Reference
This project was inspired by Sunkyoo's OpenCV4CVML Blog. The examples and ideas from the blog were instrumental in the implementation.
