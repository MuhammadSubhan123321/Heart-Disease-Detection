❤️ **Heart Disease Prediction using Machine Learning**

**This project predicts the likelihood of heart disease using Logistic Regression trained on the UCI Heart Disease dataset.
It leverages scikit-learn for model training, testing, and evaluation, achieving strong accuracy on both training and testing data.**

📌 **Features**

- Loads and processes the Heart Disease dataset (CSV)

- Splits the data into Training and Testing sets

- Trains a Logistic Regression model

- Evaluates performance using accuracy scores

- Predicts whether a person has heart disease or not based on input data

📂 **Dataset**

- File: heart.csv

- Total Records: 1025

- Columns: 14 (13 features + 1 target)

**Columns:**

- age: Age of the person

- sex: Gender (1 = male, 0 = female)

- cp: Chest pain type

- trestbps: Resting blood pressure

- chol: Cholesterol level

- fbs: Fasting blood sugar

- restecg: Resting ECG results

- thalach: Maximum heart rate achieved

- exang: Exercise induced angina

- oldpeak: ST depression induced by exercise

- slope: Slope of peak exercise ST segment

- ca: Number of major vessels colored by fluoroscopy

- thal: Thalassemia

-target: (1 = Heart Disease, 0 = Healthy)

⚙️ **Installation**

**Clone this repository and install required libraries:**

- git clone https://github.com/your-username/heart-disease-prediction.git
  cd heart-disease-prediction

- pip install -r requirements.txt

**Requirements:**

- Python 3.x

- numpy

- pandas

- scikit-learn

**You can install dependencies manually:**

- pip install numpy pandas scikit-learn

🚀 **Usage**

- Run the Jupyter Notebook or Python script to train and test the model:

- jupyter notebook Heart_Disease_Prediction.ipynb

- Or directly run the script (if saved as .py):

- python heart_disease.py

📊 **Model Training**

- Model used: Logistic Regression

- Train/Test Split: 80/20

**Accuracy:**

- Training Data: ~86.8%

- Testing Data: ~82.9%

🔍**Example Prediction**

**You can pass a patient’s health record (13 features) into the model to predict:**

- input_data = (53,1,2,130,246,1,0,173,0,0,2,3,2)
- input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
- prediction = model.predict(input_data_as_numpy_array)
______________________________________________________
if prediction[0] == 0:
    print("The person has a healthy heart 🙂")
else:
    print("The person has heart disease 💔")
______________________________________________________
📈 **Results**

✅ Balanced dataset (526 with heart disease, 499 without)
✅ Good accuracy with Logistic Regression
✅ Easily extendable to other models (Random Forest, SVM, etc.)

🏗️ **Future Improvements**

Try other ML models (Random Forest, SVM, XGBoost)

Add feature scaling & hyperparameter tuning

Deploy as a Flask/Django web app

Integrate with Streamlit for interactive predictions

👨‍💻 **Developer**

Muhammad Subhan – Full-Stack Developer (in progress)
