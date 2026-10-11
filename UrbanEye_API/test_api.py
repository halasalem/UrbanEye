import requests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "test_image.png")

with open(image_path, "rb") as f:
    response = requests.post(
        "http://127.0.0.1:5000/analyze_file",
        files={"image": f},
        timeout=60
    )

print("Status:", response.status_code)
print(response.text)