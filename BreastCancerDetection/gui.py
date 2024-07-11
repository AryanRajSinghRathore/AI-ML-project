import joblib
import numpy as np
import streamlit as st

loaded_model=joblib.load(open("C:/AI-ML/FinalProject/Diabetes_model.lb","rb"))

def diabetes_prediciton(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    predicition=loaded_model.predict(input_data_reshaped)
    print(predicition)
    if (predicition==0):
        return 'The person is not Diabetic'
    else:
        return 'The person is Diabetic'
    
def main():
    st.title('Diagnosis Predicition Web App')

    Pregnancies=st.text_input('Number Of Pregnancies')
    Gulcose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    Age=st.text_input('Age Of the person')

    diagnosis=''

    if st.button('Test Result'):
        diagnosis=diabetes_prediciton([Pregnancies,Gulcose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ =="__main__":
    main()
