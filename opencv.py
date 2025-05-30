import cv2
import numpy as np

def enhance_image(image_path):
    # Зчитуємо зображення за допомогою OpenCV
    image = cv2.imread(image_path)

    # Покращуємо деталі зображення
    enhanced_image = cv2.detailEnhance(cv2.imread(image_path), sigma_s=10, sigma_r=0.15)

    return enhanced_image

# Приклад використання
image_path = './train/addition image/product-tanker.jpg'
enhanced_image = enhance_image(image_path)

# Показуємо оригінальне та покращене зображення
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
