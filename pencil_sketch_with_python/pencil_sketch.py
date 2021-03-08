import cv2
import matplotlib.pyplot as plt
image = cv2.imread("dog.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

inverted_image = 255 - gray_image

blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
inverted_blurred = 255 - blurred

pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

plt.figure(figsize=(30,30))
plt.subplot(151), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(152), plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)),plt.title('Gray image')
plt.xticks([]), plt.yticks([])
plt.subplot(153), plt.imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB)),plt.title('Inverted image')
plt.xticks([]), plt.yticks([])
plt.subplot(154), plt.imshow(cv2.cvtColor(inverted_blurred, cv2.COLOR_BGR2RGB)),plt.title('Inverted Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(155), plt.imshow(cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB)),plt.title('Pencil filter')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
cv2.imwrite("dog_pencil_sketch.png", pencil_sketch)
print('Successfully saved')
