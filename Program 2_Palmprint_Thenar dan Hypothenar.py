import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gambar
image = cv2.imread('tangan_3.jpg', cv2.IMREAD_GRAYSCALE)

# Histogram equalization 
equalized_image = cv2.equalizeHist(image)

# Gaussian blur (Menghilangkan Noise)
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# Edge detection (Garis tangan)
edges = cv2.Canny(blurred_image, 50, 150)

# Morphological operations (Menghilangkan Tepi)
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
cleaned_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

# Mencari kontur garis tangan
contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menggambar kontur garis
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Lokasi Thenar dan Hypothenar
height, width = image.shape
thenar_region = contour_image[int(height*0.6):height, 0:int(width*0.4)]
hypothenar_region = contour_image[int(height*0.6):height, int(width*0.6):width]

# Tampilan
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.title('Original Image'), plt.imshow(image, cmap='gray')
plt.subplot(2, 3, 2), plt.title('Equalized Image'), plt.imshow(equalized_image, cmap='gray')
plt.subplot(2, 3, 3), plt.title('Edges'), plt.imshow(edges, cmap='gray')
plt.subplot(2, 3, 4), plt.title('Cleaned Edges'), plt.imshow(cleaned_edges, cmap='gray')
plt.subplot(2, 3, 5), plt.title('Contours'), plt.imshow(contour_image)
plt.subplot(2, 3, 6), plt.title('Thenar and Hypothenar Regions'), plt.imshow(np.hstack((thenar_region, hypothenar_region)))

plt.tight_layout()
plt.show()
