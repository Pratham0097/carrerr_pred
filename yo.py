import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load model + scaler
# -----------------------------
model = pickle.load(open("final_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Career labels (same order used during training)
career_labels = [
    "AI/ML",
    "Cloud Computing",
    "Cyber Security",
    "Data Science",
    "UI/UX",
    "Web Development"
]

# -----------------------------
# UI Design
# -----------------------------
st.set_page_config(page_title="Career Recommender", page_icon="🎯")

st.title("🎯 Multi-Class Career Recommendation System")
st.write("Rate your skills from 1 (low) to 10 (high)")

# -----------------------------
# User Inputs
# -----------------------------
python = st.slider("Python Skill",1,10,5)
math = st.slider("Math Skill",1,10,5)
logic = st.slider("Logical Thinking",1,10,5)
design = st.slider("Design Creativity",1,10,5)
networking = st.slider("Networking Knowledge",1,10,5)
problem = st.slider("Problem Solving",1,10,5)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Recommend Career"):

    # create input array
    user_data = np.array([[python, math, logic, design, networking, problem]])

    # scale input
    user_scaled = scaler.transform(user_data)

    # prediction
    prediction = model.predict(user_scaled)[0]

    # probabilities
    probs = model.predict_proba(user_scaled)[0]

    # sort top predictions
    top_indices = np.argsort(probs)[::-1]

    # -----------------------------
    # Display Results
    # -----------------------------
    st.success(f"Recommended Career: {career_labels[prediction]}")

    st.subheader("Top Career Matches")

    for i in top_indices[:3]:
        st.write(f"{career_labels[i]} → {probs[i]*100:.2f}%")

    # Bar chart visualization
    st.subheader("Prediction Confidence")
    st.bar_chart(probs)