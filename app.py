
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Web-page Settings
st.set_page_config(page_title="Wine Classifier Pro", page_icon="🍷", layout="wide")
st.title("🍷 Wine-Classification App")
st.write("Select a method to predict wine varieties.")

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load("best_pipeline_wine_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'best_pipeline_wine_model.joblib' not found. Please train and save the model first!")
    st.stop()

# Class-names for better display
class_names = {0: "Class 0 (Barolo-Typ)", 1: "Class 1 (Grignolino-Typ)", 2: "Class 2 (Barbera-Typ)"}

# 3. Modus selection in the sidebar
modus = st.sidebar.radio("Select Modus:", ["Single Wine (sliders)", "Batch-Prediction (CSV Upload)"])

# ==========================================
# MODUS 1: SINGLE PREDICTION
# ==========================================
if modus == "Single Wine (sliders)":
    st.header("🔬 Single analysis via slider")
    
    # Input fields for the 4 key characteristics
    # Characteristics: alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,proline
    alcohol = st.slider("Alcohol content (Alcohol)", 11.0, 15.0, 13.0, 0.1)
    magnesium = st.slider("Magnesium", 70.0, 160.0, 100.0, 1.0)
    flavanoids = st.slider("Flavanoids", 0.3, 5.0, 2.0, 0.1)
    proline = st.slider("Proline", 270.0, 1680.0, 750.0, 10.0)

    # Fill the remaining 9 features with average values
    features = np.array([[
        alcohol, 2.3, 2.3, 19.5, magnesium, 2.3, flavanoids, 
        0.3, 1.6, 5.0, 0.9, 2.6, proline
    ]])

    if st.button("Predict wine type", type="primary"):
        prediction = model.predict(features)[0]
        st.success(f"The Model predicts: *{class_names.get(prediction)}*")

# ==========================================
# MODUS 2: CSV-UPLOAD
# ==========================================
else:
    st.header("📊 Batch prediction via CSV-file")
    st.write("The uploaded CSV file must contain exactly the *13 columns (features)* of the wine dataset in the correct order.")
    
    # Provide a template download for the user
    example_data = pd.DataFrame([[13.0, 2.3, 2.3, 19.5, 100.0, 2.3, 2.0, 0.3, 1.6, 5.0, 0.9, 2.6, 750.0]], 
                                  columns=[f"Feature_{i}" for i in range(1, 14)])
    st.download_button(
        label="📄 Download sample-CSV",
        data=example_data.to_csv(index=False),
        file_name="wine_example.csv",
        mime="text/csv"
    )

    # File Uploader
    uploaded_file = st.file_uploader("Select a CSV-file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded data (preview):", df.head())
            
            # Security check of the number of columns
            if df.shape[1] != 13:
                st.error(f"Error: The file has {df.shape[1]} columns, but the model requires exacty 13 columns!")
            else:
                if st.button("Start prediction for all rows", type="primary"):
                    # Prediction across all rows (pipeline automatically scales each row)
                    predictions = model.predict(df)
                    
                    # Convert results to readable text and append as a new column
                    df["Predicted_Class_ID"] = predictions
                    df["Wine_Type_Name"] = [class_names.get(p) for p in predictions]
                    
                    st.success("🎉 Prediction completed successfully!")
                    st.write("### Results:", df[["Wine_Type_Name"] + list(df.columns[:-2])])
                    
                    # Make CSV-result available for download
                    result_csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=result_csv,
                        file_name="wine_prediction_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing the file: {e}")
