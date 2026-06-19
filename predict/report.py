import requests

url = "http://127.0.0.1:8000/predict/report"
file_path = "/mnt/workspace/BrainCT/datasets_filtered/CT/CQ500-CT-5_CT000011.dcm"

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {
        "case_id": "TEST001",
        "patient_name": "张三",
        "patient_age": "45"
    }
    response = requests.post(url, files=files, data=data)

print("状态码:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("\n生成的报告:\n")
    print(result["report"])
else:
    print("错误:", response.text)