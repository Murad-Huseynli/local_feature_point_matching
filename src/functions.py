import cv2
import numpy as np
from skimage.feature import peak_local_max

# Global Parameters for tuning (already tuned) 
a = 0.04
params1 = [(0, 0), 0.14]
params2 = [(0, 0), 0.162]
params3 = [(0, 0), 0.293]
corner_response_threshold = 0.6 # or 0.05 both yielded to good results of surpassing 80% as required
descriptor_clipping_threshold = 0.1
nndr_ratio_threshold = 0.9


def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    m, n = image.shape # Getting the shape of the processed image 
    R = np.zeros((m, n)) # Creating a Hariss respond matrix to store the respond values of each pixel
    w = descriptor_window_image_width // 2 # Utilizing w further for implicit boundary suppression 

    # Computing gradients for future usage in second moment matrix M and corner response calculation 

    # Pre-processing applying gaussian blur to get better results later on
    image = cv2.GaussianBlur(image, *params1)

    # Step 1 - Computing gradients Ix and Iy
    # Computing the gradient in vertical-x direction
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3) # similar to cv2.filter2D() with vertical sobel = [1,0,-1], [2,0,-2], [1,0,-1]
    # Computing the gradient in horizontal-y direction
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3) # similar to cv2.filter2D() with horizontal sobel = [1,2,1], [0,0,0], [-1,-2,-1]

    # Step 2 - Computing products of derivatives at each pixel
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Applying Gaussian filter to smooth the squares of derivatives
    Sxx = cv2.GaussianBlur(Ixx, *params1)
    Syy = cv2.GaussianBlur(Iyy, *params2)
    Sxy = cv2.GaussianBlur(Ixy, *params3)

    # Step 3 - Computing the Harris corner response for each pixel using vectorized approach - could be possibly used, as well
    # R = (Ixx * Iyy - Ixy ** 2) - a * (Ixx + Iyy) ** 2
    
    # Step 3 - Computing cornerness (corner response) of each pixel + implicit border suppression 
    for i in range(w, m - w + 1):
        for j in range(w, n - w + 1):
            # Calculating the elements of second moment matrix (M) using the windowing approach
            t_gXX = np.sum(Sxx[i - w : i + w, j - w : j + w]) 
            t_gYY = np.sum(Syy[i - w : i + w, j - w : j + w])
            t_gXY = np.sum(Sxy[i - w : i + w, j - w : j + w])
            
            # Computing the Harris corner response (c) for each pixel using the formula from the lectures (with the det(M), trace(M) , and alpha-value)
            c = t_gXX * t_gYY - t_gXY ** 2 - a * (t_gXX + t_gYY) ** 2
            R[i, j] = c

    # Step 4 - Thresholding on C to pick high cornerness  
    C_thresholded = np.where(R > corner_response_threshold, R, 0)

    # Step 5 - Applyting non-maxima suppression to pick the peaks 
    coordinates = peak_local_max(C_thresholded, min_distance=3, threshold_abs=corner_response_threshold)

    # Retrieving y and x coordinates (arrays) from coordinates (n x 2) array
    y, x = coordinates[:, 0], coordinates[:, 1]
    
    return x, y

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000


def get_descriptors(image, x, y, descriptor_window_image_width):
     # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)

    # Extracting the number of keypoints to find the descriptors for 
    numOfKeypoints = len(x)
    half_window_size = descriptor_window_image_width // 2
    cell_width = descriptor_window_image_width // 4
    # Typical SIFT-like descriptor usually has a 4x4 grid of cells and 8 orientation bins per cell
    feature_dimensionality = 4 * 4 * 8

    # As per the requirements, initiliazing the 'features' array of dimensions [len(x) * feature dimensionality]
    features = np.zeros((numOfKeypoints, feature_dimensionality))
    
    # GaussianBlur on the entire image for getting better results in the further stages 
    image_blurred = cv2.GaussianBlur(image, *params2)

    # Computing the gradients using Sobel convolution operator 

    # Computing the gradient in vertical-x direction
    Ix = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3) # similar to cv2.filter2D() with vertical sobel = [1,0,-1], [2,0,-2], [1,0,-1]
    # Computing the gradient in horizontal-y direction
    Iy = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3) #  similar to cv2.filter2D() with horizontal sobel = [1,2,1], [0,0,0], [-1,-2,-1]
    
    # Computing the magnitude and orientation of each pixel

    #magnitude, orientation = cv2.cartToPolar(Ix, Iy, angleInDegrees=False)

    orientation = np.arctan2(Iy, Ix)
    
    #Enforcing the directions to be in the valid region of [0; 2pi]
    orientation = np.where(orientation >= 0, orientation, orientation + 2 * np.pi)

    magnitude = np.sqrt(Ix**2 + Iy**2)

    # Iterating over each keypoint to compute its descriptor
    for i in range(numOfKeypoints):
        
        # Initializing an empty list to store a descriptor for the current keypoint
        descriptor = []

        # # Enforcing border suppression 
        # if (int(x[i]) < half_window_size or
        # int(x[i]) > image.shape[1] - half_window_size or
        # int(y[i]) < half_window_size or
        # int(y[i]) > image.shape[0] - half_window_size):
        #     continue
        
        # Defining the window around the keypoint + implicit border suppression 
        top_left_x = max(int(x[i]) - half_window_size, 0)
        top_left_y = max(int(y[i]) - half_window_size, 0)
        bottom_right_x = min(int(x[i]) + half_window_size, image.shape[1])
        bottom_right_y = min(int(y[i]) + half_window_size, image.shape[0])

        # Extracting the window (patch) of the gradient magnitude and orientation
        magnitude_patch = magnitude[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        orientation_patch = orientation[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # Blurring the windows of the gradient magnitude and orientation for better results
        magnitude_patch = cv2.GaussianBlur(magnitude_patch, *params3)
        orientation_patch = cv2.GaussianBlur(orientation_patch, *params3)
        
        # Dividing the window into a 4x4 grid of cells and computing histogram for each cell using list comprehension
        hist = [
    np.histogram(
        orientation_patch[i * cell_width:(i + 1) * cell_width, j * cell_width:(j + 1) * cell_width],
        bins=8,
        range=(0, 2 * np.pi),
        weights=magnitude_patch[i * cell_width:(i + 1) * cell_width, j * cell_width:(j + 1) * cell_width]
    )[0]
    for i in range(cell_width) for j in range(cell_width)
]
        descriptor = np.concatenate(hist)
        
        # Normalizing the descriptor to unit length (adding a very small value to avoid division by 0)
        descriptor = np.array(descriptor) / (np.linalg.norm(descriptor) + 1e-50)

        # Thresholding (clipping) to gain illuminance invariance
        descriptor = np.minimum(descriptor, descriptor_clipping_threshold)

        # Renormalizing the descriptor back (adding a very small value to avoid division by 0)
        descriptor /= (np.linalg.norm(descriptor) + 1e-50)

        # Saving the descriptor 
        features[i, :] = descriptor

    return features


def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.

    '''
    Retrieving the number of feature vectors from features 1, since the implementation of the function is asymmetric, we can use f2 = features2.shape[0] to retrieve the number of features in features2 and proceed that way,as well (which may produce a different result, therefore, usually, it is better to use a more accurate, better quality image as the reference one)
    '''
    f1 = features1.shape[0]
    #f2 = features2.shape[0]

    # Initializing empty lists for storing matches and the confidence levels
    matches = []
    confidences = []

    for i in range(f1):
        # Picking the i-th vector from features1 to find its best match among features2 vectors
        v1 = features1[i] 
        # Finding the difference of corresponding coordinates in the feature space between v1 and all features2 vectors using vectorized, broadcasting operation for improving complexity and efficiency
        differences = features2 - v1
        # Finding L2 norm for each difference vector from matrix "differences", leads to the calculation of Euclidean Distance between v1 and every feature in features2
        normalized_differences = np.linalg.norm(differences, axis=1)
        # Sorting the indexes of the distances in the ascending order to find the first and second closest neighbors
        nearest_two = np.argsort(normalized_differences)
        # Retrieving the distances from v1 to first and second nearest neighbors and calculating the ratio as per the NNDR formula (adding a )
        nndr_ratio = normalized_differences[nearest_two[0]] / (normalized_differences[nearest_two[1]] + 1e-50)
        # Filtering the nearest neighbor based on the threshold (checking if it is a good match)
        if nndr_ratio < nndr_ratio_threshold: 
            # Appending the indexes of v1 from features1 and its best match from features2 (nearest_two[0])
            matches.append([i, nearest_two[0]])
            # Calculating and saving the confidence levels
            confidences.append(1 - nndr_ratio)

    # List to numpy array conversion for consistent data handling    
    matches = np.array(matches)
    confidences = np.array(confidences)

    return matches, confidences


# Normalized patches-based, naive descriptor

# def get_descriptors(image, x, y, descriptor_window_image_width):
#      # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
#     # Revised Python codes are written by Inseung Hwang at KAIST.

#     # Returns a set of feature descriptors for a given set of interest points.

#     # 'image' can be grayscale or color, your choice.
#     # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
#     #   The local features should be centered at x and y.
#     # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
#     #   You can assume that descriptor_window_image_width will be a multiple of 4
#     #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
#     # If you want to detect and describe features at multiple scales or
#     # particular orientations, then you can add input arguments.

#     # 'features' is the array of computed features. It should have the
#     #   following size: [length(x) x feature dimensionality] (e.g. 128 for
#     #   standard SIFT)

#     # Extracting the number of keypoints to find the descriptors for 
#     numOfKeypoints = len(x)
#     # Each descriptor will have a size of descriptor_window_image_width squared
#     feature_dimensionality = descriptor_window_image_width ** 2
#     features = np.zeros((numOfKeypoints, feature_dimensionality))
    
#     half_window_size = descriptor_window_image_width // 2

#     for i in range(numOfKeypoints):
#         top_left_y = int(max(y[i] - half_window_size, 0))
#         bottom_right_y = int(min(y[i] + half_window_size, image.shape[0]))
#         top_left_x = int(max(x[i] - half_window_size, 0))
#         bottom_right_x = int(min(x[i] + half_window_size, image.shape[1]))
        
#         # Extracting the patch
#         patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].astype(np.float32)

#         # Checking if the patch size is as expected; if not, skipping this keypoint
#         if patch.shape[0] != descriptor_window_image_width or patch.shape[1] != descriptor_window_image_width:
#             continue

#         # Normalizing the patch
#         patch -= np.mean(patch)
#         patch_std = np.std(patch)
#         if patch_std > 1e-50:  # Avoiding division by zero
#             patch /= patch_std
        
#         # Flattening and storing the normalized patch as the descriptor
#         features[i, :] = patch.flatten()

#     return features
