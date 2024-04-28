from flask import Flask, request, jsonify, render_template
import pickle
## Load the model
input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/', methods=['GET']) # Homepage
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML template
    """
    # customer = request.get_json()
    customer = request.form.to_dict()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5  # Assuming a threshold of 0.5 for churn prediction
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    # return jsonify(result)
    return result
    return render_template('index.html', prediction_results = result)


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True)