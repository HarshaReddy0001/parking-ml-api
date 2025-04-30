import requests

res = requests.post("http://127.0.0.1:8000/predict", json={
    "message": "Can I get a parking slot at 6PM on Saturday?"
})
print(res.json())
