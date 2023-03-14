# import cv2
# import numpy as np
# import requests
# import base64

# # URL of the API endpoint
# url = "http://172.16.100.102:8000/"

# # API 끝점에 HTTP GET 요청 보내기
# response = requests.get(url)

# # base64 인코딩된 이미지 데이터를 가져오기
# data = response.json()["data"]

# # Base64 인코딩 영상 데이터를 바이트 배열로 디코딩
# img_bytes = base64.b64decode(data)

# # 바이트 배열을 널피 배열로 변환
# img_np = np.frombuffer(img_bytes, dtype=np.uint8)

# # Decode the numpy array as an image using OpenCV
# img = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

# # Display the image using OpenCV
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import requests
import base64
import time

url = "http://172.16.100.102:8000/"

response = requests.get(url)
data = response.json()["data"]

img_bytes = base64.b64decode(data)

img_np = np.frombuffer(img_bytes, dtype=np.uint8)

img = cv2.imdecode(img_np, flags=cv2.IMREAD_COLOR)

start_time = time.time()
cv2.imshow("Image", img)
end_time = time.time()
time_taken = end_time - start_time
fps = 1.0 / time_taken

font = cv2.FONT_HERSHEY_SIMPLEX
text = f"FPS: {fps:.2f}"
cv2.putText(img, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Image with FPS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()