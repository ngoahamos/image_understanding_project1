import cv2
import numpy as np
import csv


# Step 2: Warp images to spherical/cylindrical coordinates and estimate focal length
def warp_to_cylindrical(images, focal_length=None):
    if focal_length is None:
        print("Focal length estimation")
        image0 = images[0]
        image1 = images[1]

        print("keypoints and descriptors between the first two images")
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image0, None)
        kp2, des2 = sift.detectAndCompute(image1, None)

        print("FLANN-based matcher")
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors using FLANN
        print("About to perform descriptor matching ...")
        matches = flann.knnMatch(des1, des2, k=2)
        print("✅ Matches obtained")

        print("Perform Lowe's ratio test to get good matches")
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])

        print("Perform RANSAC to estimate fundamental matrix and focal length")
        fundamental_matrix, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
        focal_length = 1 / fundamental_matrix[0, 0]
        print("✅ Estimated Focal Length {}".format(focal_length))

    warped_images = []

    for image in images:
        height, width = image.shape[:2]
        cylindrical_image = np.zeros_like(image)

        for x in range(width):
            for y in range(height):
                theta = (x - width / 2) / focal_length
                h = (y - height / 2) / focal_length
                x_cart = focal_length * np.tan(theta)
                y_cart = focal_length * h
                x_new = int(x_cart + width / 2)
                y_new = int(y_cart + height / 2)

                if 0 <= x_new < width and 0 <= y_new < height:
                    cylindrical_image[y, x] = image[y_new, x_new]

        warped_images.append(cylindrical_image)

    return warped_images, focal_length
   
# Step 3: Extract features
def extract_features(images):
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []

    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(des)

    return keypoints, descriptors

# Step 4: Align neighboring pairs using RANSAC
def align_images_ransac(images, keypoints, descriptors):
    aligned_images = []
    translations = []

    for i in range(len(images) - 1):
        kp1, kp2 = keypoints[i], keypoints[i + 1]
        des1, des2 = descriptors[i], descriptors[i + 1]

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            aligned_images.append(images[i])
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(images[i], M, (images[i].shape[1], images[i].shape[0]))

        aligned_images.append(aligned_image)
        translations.append(("image" + str(i), "image" + str(i + 1), M[0, 2], M[1, 2]))

    return aligned_images, translations

# Step 5: Write out list of neighboring translations
def write_translation(translations, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image1", "Image2", "TranslationX", "TranslationY"])
        for translation in translations:
            csvwriter.writerow([translation[0], translation[1], translation[2], translation[3]])

# Step 6: Correct for drift (implement your drift correction algorithm)
def correct_drift(images, translations):
    corrected_images = [images[0]]  # The first image remains unchanged

    for i in range(1, len(images)):
        translation_x_accumulated = 0
        translation_y_accumulated = 0

        for j in range(i):
            translation_x_accumulated += translations[j][2]
            translation_y_accumulated += translations[j][3]

        translation_matrix = np.float32([[1, 0, translation_x_accumulated], [0, 1, translation_y_accumulated]])
        drifted_image = cv2.warpAffine(images[i], translation_matrix, (images[i].shape[1], images[i].shape[0]))
        corrected_images.append(drifted_image)

    return corrected_images


# Step 7: Read in warped images and blend them
def blend_warped_images(warped_images):
    blended_image = np.zeros_like(warped_images[0])

    for image in warped_images:
        blended_image += image

    # blended_image /= (len(warped_images) * 1.0)

    return blended_image

# Step 8: Crop the result and import into a viewer
def crop_and_view(result_image):
    # Implement cropping and displaying the result in a viewer
    # Use OpenCV or a similar library for image manipulation and display

    cv2.imshow("Blended Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_paths = ['image0.JPG', 'image1.JPG', 'image2.JPG']
print('about to read images');
images = [cv2.imread(path) for path in image_paths]
print("✅ Done reading images");

print("about to warp images")
warped_images, focal_length = warp_to_cylindrical(images)
print("✅ Done warping images")

print("Extracting features")
keypoints, descriptors = extract_features(warped_images)
print("✅ Features Extracted")

print("Align neighboring pairs using RANSAC")
aligned_images, translations = align_images_ransac(warped_images, keypoints, descriptors)
print("✅ Done Aligning images")

print("list of neighboring translations")
write_translation(translations, 'translations.csv')
print("✅ translation saved")

# Continue with other steps, correct for drift, blend images, crop, and view the result
print("let's correct drift")
corrected_images = correct_drift(aligned_images, translations)
print("✅ Image corrected")

print("Let's create panaroma")
blended_image = blend_warped_images(corrected_images)
print("✅ panaroma created");

print("Let's show result")
crop_and_view(blended_image)