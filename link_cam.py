import json
def load_link():
    file_path = "config/cam.json"

    # Đọc tệp tin JSON
    with open(file_path) as file:
        data = json.load(file)
        # print(data)
    return data