# Importing Libraries
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import streamlit as st

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

# Reading the training data
df = pd.read_csv("E:\python\Disease_Prediction_Training.csv")
df.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

# Reading the testing data
tr = pd.read_csv("E:\python\Disease_Prediction_Testing.csv")
tr.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

# Defining feature and target variables
X = df[l1]
y = df["prognosis"]

X_test = tr[l1]
y_test = tr["prognosis"]

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X, y)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X, y)

# Accuracy of the classifiers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

y_pred_dt = clf_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Random Forest Classifier")
print("Accuracy:", accuracy_rf)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("Decision Tree Classifier")
print("Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Streamlit user interface
def get_symptoms_and_predict():
    st.title("Disease Prediction System")
    st.write("Please select your symptoms")

    algorithm = st.selectbox("Select Algorithm", ["Random Forest", "Decision Tree"])
    psymptoms = st.multiselect("Symptoms", l1)

    l2 = [0] * len(l1)
    for symptom in psymptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1

    inputtest = [l2]
    if st.button("Predict"):
        if algorithm == "Random Forest":
            predict = clf_rf.predict(inputtest)
            predicted = predict[0]
            accuracy = accuracy_rf
        else:
            predict = clf_dt.predict(inputtest)
            predicted = predict[0]
            accuracy = accuracy_dt

        if predicted in range(len(disease)):
            st.success(f"The predicted disease is: {disease[predicted]}")
        else:
            st.error("Disease not found")

        st.markdown(f"<h3 style='text-align: center; color: white ;'>Model Accuracy: {accuracy:.2f}</h3>", unsafe_allow_html=True)

# Main program execution
if __name__ == "__main__":
    get_symptoms_and_predict()

