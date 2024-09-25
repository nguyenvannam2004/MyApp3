from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  
from flask_cors import CORS
from model_nerualnetwork import NeuralNetwork

app = Flask(__name__)
CORS(app)

# Tải các mô hình đã lưu
model1 = joblib.load('logistic_model.pkl')
model2 = joblib.load('neuralnetwork_model.pkl')
model3 = joblib.load('ensemble_model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_model = data.get('model')

    if selected_model == 'model1':
        model = model1
    elif selected_model == 'model2':
        model = model2
    elif selected_model == 'voting_classifier':
        model = model3
    else:
        return jsonify({"message": "Mô hình không hợp lệ"}), 400
    
    try:
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"message": "Dữ liệu đầu vào không hợp lệ", "error": str(e)}), 400

    print("Features:", features)

    # Chuyển đổi thành DataFrame với tên cột
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 
                     'slope', 'ca', 'thal']
    features_df = pd.DataFrame([features], columns=feature_names)

    # Dự đoán bằng mô hình
    prediction = model.predict(features_df)

    result = "Có nguy cơ mắc bệnh tim\n\n\n\n Đừng buồn, bạn hãy chạy vào trong vườn hái một quả chanh.\nBổ đôi nó ra(nhớ là phải bổ ngang nha) xong vắt nó vào cốc, cho 2 thìa đường,500ml nước đun sôi để nguội\nCho thêm 2 viên đá nữa cho mát rồi khuấy đều lên sẽ thu đc dung dịch hay còn gọi là nước đường.\nUống nó! Nó sẽ ko giúp bạn hết bị bệnh tim đâu nhưng mà nươc đường thì rất ngọt, với lại biết đâu đó có thể sẽ là lần cuối cùng mà bạn đc uống nước đường thì sao =))))" if prediction[0] == 1 else "Không có nguy cơ mắc bệnh tim"
    
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)
