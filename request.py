import request

url = 'http://localhost:5000/predict_api'
r = request.post(url,json={'tv':5000, 'radio':1000, 'newspaper':100})

print(r.json())