import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Set up the page
st.set_page_config(page_title="CareerGPT", page_icon="🤖")
st.title("🤖 Career Guidance Chatbot")
st.markdown("Fill in your details to get personalized career recommendations!")

# Create input form with multiple fields
with st.form("career_form"):
    st.subheader("Your Profile")
    skills = st.text_input("Skills (comma-separated):", "Python, SQL, Data Analysis")
    career = st.text_input("Desired Career Path:", "Data Scientist")
    industry = st.text_input("Preferred Industry:", "IT-Software")
    experience = st.selectbox("Years of Experience:", ["0-1 yrs", "2-5 yrs", "5+ yrs"])
    salary = st.text_input("Expected Salary Range:", "5,00,000 - 10,00,000 PA")
    
    submitted = st.form_submit_button("Get Recommendation")

# When form is submitted
if submitted:
    try:
        # Load model
        with open('career_recommender.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Combine all inputs into a single string (matching training data format)
        user_input = f"{skills} {career} {industry} {experience} {salary}"
        
        # Get prediction
        prediction = model.predict([user_input])[0]
        
        # Display results
        st.success(f"**Recommended Career:** {prediction}")
        st.markdown("---")
        st.subheader("💡 Career Development Tips")
        st.write(f"- Strengthen your {prediction.split()[0]} skills with advanced courses")
        st.write("- Network with professionals in this field on LinkedIn")
        st.write(f"- Research growing companies in {industry} sector")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure 'career_recommender.pkl' exists in your directory")