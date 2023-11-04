import cv2
import numpy as np

# Step 2: Warp images to cylindrical coordinates and estimate focal length
def warp_to_cylindrical(images):
    focal_length = 1000.0  # Adjust as needed

    warped_images = []
    for image in images:
        height, width = image.shape[:2]
        cylindrical_image = np.zeros_like(image, dtype=np.uint8)

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
        image1, image2 = images[i], images[i + 1]
        kp1, kp2 = keypoints[i], keypoints[i + 1]
        des1, des2 = descriptors[i], descriptors[i + 1]

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            aligned_images.append(image1)
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(image1, M, (image1.shape[1], image1.shape[0]))

        aligned_images.append(aligned_image)
        translations.append((f"image{i}", f"image{i + 1}", M[0, 2], M[1, 2]))

    return aligned_images, translations

# Step 5: Write out list of neighboring translations
def write_translation(translations, output_file):
    with open(output_file, 'w') as file:
        for translation in translations:
            file.write(f"{translation[0]} -> {translation[1]}: X: {translation[2]}, Y: {translation[3]}\n")

# Step 6: Correct for drift (implement your drift correction algorithm)

# Step 7: Read in warped images and blend them
def blend_warped_images(warped_images):
    blended_image = np.zeros_like(warped_images[0], dtype=np.float64)

    for image in warped_images:
        blended_image += image

    blended_image /= len(warped_images)

    return blended_image

# Step 8: Crop the result and import into a viewer
def crop_and_view(result_image):
    # Implement cropping and displaying the result in a viewer
    # Use OpenCV or a similar library for image manipulation and display

    result_image = result_image.astype(np.uint8)
    cv2.imshow("Panorama", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_paths = ['image0.JPG', 'image1.JPG', 'image2.JPG']
images = [cv2.imread(path) for path in image_paths]

# Step 2: Warp images to cylindrical coordinates and estimate focal length
warped_images, focal_length = warp_to_cylindrical(images)

# Step 3: Extract features
keypoints, descriptors = extract_features(warped_images)

# Step 4: Align neighboring pairs using RANSAC
aligned_images, translations = align_images_ransac(warped_images, keypoints, descriptors)

# Step 5: Write out list of neighboring translations
write_translation(translations, 'translations.txt')

# Step 7: Read in warped images and blend them
blended_image = blend_warped_images(aligned_images)

# Step 8: Crop and view the result
crop_and_view(blended_image)