from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ------------------------------
# Load trained VQC model
with open('vqc_model.pkl', 'rb') as f:
    vqc_model = pickle.load(f)

# ------------------------------
# Home page
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# Predict page with manual input + graph
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    graph_url = None

    if request.method == 'POST':
        # Read features from form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Prediction
        prediction = vqc_model.predict([[feature1, feature2]])[0]

        # Generate graph
        plt.figure(figsize=(5,4))
        plt.bar(['Prediction'], [prediction], color='blue')
        plt.title('Prediction Result')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

    return render_template('predict.html', prediction=prediction, graph_url=graph_url)

# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ------------------------------
# Load trained VQC model
with open('vqc_model.pkl', 'rb') as f:
    vqc_model = pickle.load(f)

# ------------------------------
# Home page
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# Predict page with manual input + graph
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    graph_url = None

    if request.method == 'POST':
        # Read features from form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])

        # Prediction
        prediction = vqc_model.predict([[feature1, feature2]])[0]

        # Generate graph
        plt.figure(figsize=(5,4))
        plt.bar(['Prediction'], [prediction], color='blue')
        plt.title('Prediction Result')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

    return render_template('predict.html', prediction=prediction, graph_url=graph_url)

# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)



