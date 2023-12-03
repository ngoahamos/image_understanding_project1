import cv2

# Load the image
image_path = 'group-scale.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=13) # 1.3 | 13

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save or display the result
cv2.imshow('Detected Faces', image)
cv2.imwrite('group_scale_detected.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()