import requests

url = "http://127.0.0.1:8000/predict/volume"
zip_path = "/mnt/workspace/BrainCT/predict/CQ500-CT-5.zip"  # 自行打包

with open(zip_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())