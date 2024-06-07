import json
import requests


crop_api_url = "http://localhost:8000"


def test_pytorch_predict_status():
    url = f"{crop_api_url}/pytorch_predict"
    path = 'C:/Users/luisg/Documents/projects/cifar-CNN/data/' + \
        'Agricultural-crops-new/test/banana/image (15).jpg'
    payload = {}
    files = [
        ('file', ('image (15).jpg', open(path, 'rb'), 'image/jpeg'))
    ]
    headers = {
        # 'Content-Type': 'multipart/form-data',
        'Accept': 'application/json'
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files
    )
    details = json.loads(response.text)

    assert response.status_code == 200
    assert details['status'] == "success"


def test_pytorch_predict_negative_score():
    url = f"{crop_api_url}/pytorch_predict"
    path = 'C:/Users/luisg/Documents/projects/cifar-CNN/data/' + \
        'Agricultural-crops-new/test/banana/image (15).jpg'
    payload = {}
    files = [
        ('file', ('image (15).jpg', open(path, 'rb'), 'image/jpeg'))
    ]
    headers = {
        # 'Content-Type': 'multipart/form-data',
        'Accept': 'application/json'
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files
    )
    details = json.loads(response.text)

    assert details['data']['score'] > 0


def test_pytorch_predict_cherry():
    url = f"{crop_api_url}/pytorch_predict"
    path = 'C:/Users/luisg/Documents/projects/cifar-CNN/data/' + \
        'Agricultural-crops-new/test/Cherry/images20.jpg'
    payload = {}
    files = [
        ('file', ('image (15).jpg', open(path, 'rb'), 'image/jpeg'))
    ]
    headers = {
        # 'Content-Type': 'multipart/form-data',
        'Accept': 'application/json'
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files
    )
    details = json.loads(response.text)

    assert response.status_code == 200
    assert details['status'] == "success"
    assert details['error'] is None
    assert details['data']['class_name'] == "Cherry"
