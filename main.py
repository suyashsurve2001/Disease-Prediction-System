import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import requests
import io

# URLs to the CSV files on GitHub
training_url = 'https://raw.githubusercontent.com/suyashsurve2001/Disease-Prediction-System/main/Disease_Prediction_Training.csv'
testing_url = 'https://raw.githubusercontent.com/suyashsurve2001/Disease-Prediction-System/main/Disease_Prediction_Testing.csv'

# Download the CSV files
@st.cache_data
def load_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return pd.read_csv(io.StringIO(response.text))

# Load training and testing data
training_df = load_data(training_url)
testing_df = load_data(testing_url)

# Display the DataFrames
st.write("Training Data")
st.write(training_df)
st.write("Testing Data")
st.write(testing_df)

# Assuming you have some data processing and visualization code here
scaler = StandardScaler()
scaled_training_data = scaler.fit_transform(training_df.select_dtypes(include=[np.number]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scaled_training_data[:, 0], scaled_training_data[:, 1], scaled_training_data[:, 2])
st.pyplot(fig)

# Your existing machine learning code, updated to use the loaded DataFrames
# List of symptoms
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
      'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
      'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
      'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
      'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
      'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
      'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
      'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
      'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
      'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
      'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
      'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
      'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
      'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
      'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
      'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
      'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
      'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
      'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
      'yellow_crust_ooze']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
           'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
           'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
           'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
           'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
           'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
           'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
           'Osteoarthristis', 'Arthritis',
           '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
           'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Prepare the data
training_df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
testing_df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

X_train = training_df[l1]
y_train = training_df["prognosis"]
X_test = testing_df[l1]
y_test = testing_df["prognosis"]

# Random Forest Classifier  
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Accuracy of the classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = clf.predict(X_test)
st.write("Random Forest Classifier")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Streamlit user interface
def get_symptoms_and_predict():
    st.title("Disease Prediction System")
    st.write("Please select your symptoms")

    psymptoms = st.multiselect("Symptoms", l1)

    l2 = [0] * len(l1)
    for symptom in psymptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1

    inputtest = [l2]
    if st.button("Predict"):
        predict = clf.predict(inputtest)
        predicted = predict[0]

        if predicted in range(len(disease)):
            st.success(f"The predicted disease is: {disease[predicted]}")
        else:
            st.error("Disease not found")

# Main program execution
if __name__ == "__main__":
    get_symptoms_and_predict()
