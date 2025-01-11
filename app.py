from flask import Flask, request, render_template
import joblib  # or import your model

app = Flask(__name__)

# Load your trained model
model = joblib.load('breast_cancer01')

@app.route('/')
def home():
    return render_template('index01.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the user
    features = [float(request.form[f]) for f in [
        'clump_thickness',
        'cell_size',
        'cell_shape',
        'marginal_adhesion',
        'epithelial_cell_size',
        'bare_nuclei',
        'bland_chromatin',
        'normal_nucleoli',
        'mitoses'
    ]]
    
    prediction = model.predict([features])

    # Return the prediction result
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
