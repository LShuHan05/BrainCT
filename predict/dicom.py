import requests
import json

url = "http://127.0.0.1:8000/predict/dicom"
file_path = "/mnt/workspace/BrainCT/datasets_filtered/CT/CQ500-CT-5_CT000011.dcm"   # 请替换为实际路径

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("状态码:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("预测结果:")
    for pred in result["predictions"]:
        print(f"  {pred['label']}: {pred['probability']:.4f} -> {'阳性' if pred['positive'] else '阴性'}")
else:
    print("错误:", response.text)