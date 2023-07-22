import cv2
# import numpy as np
#
# def find_largest_rectangle(image_path):
#     # Read the image
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Apply Canny edge detection to find edges in the image
#     edges = cv2.Canny(img, 50, 150)
#
#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Find the largest contour (rectangle) in the image
#     max_area = 0
#     largest_rectangle = None
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > max_area:
#             max_area = area
#             largest_rectangle = contour
#
#     # Find the bounding box coordinates of the largest rectangle
#     x, y, w, h = cv2.boundingRect(largest_rectangle)
#
#     # Crop the largest rectangle from the original image
#     cropped_image = img[y:y+h, x:x+w]
#
#     return cropped_image
# import cv2
import numpy as np

def find_largest_rectangle(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges in the grayscale image
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (rectangle) in the image
    max_area = 0
    largest_rectangle = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_rectangle = contour

    # Find the bounding box coordinates of the largest rectangle
    x, y, w, h = cv2.boundingRect(largest_rectangle)

    # Crop the largest rectangle from the original image
    cropped_image = img[y:y+h, x:x+w]


    return cropped_image

# if __name__ == "__main__":
    image_path = "/mnt/c/Users/itay/Desktop/notebooks/all_graphs/graph_9.png"  # Replace this with the actual path of your image

    cropped_img = find_largest_rectangle(image_path)

    # Save the cropped image
    cv2.imwrite("/mnt/c/Users/itay/Desktop/notebooks/all_graphs/graph_9_crop.png", cropped_img)