import openai
import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import openpyxl

# Set your OpenAI API Key here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function for Sentiment Analysis of Review
def analyze_sentiment(review_text):
    prompt = f"Analyze the sentiment of the following review and classify it as positive or negative:\n\n{review_text}"
    
    response = openai.Completion.create(
        engine="gpt-4",  # Or use gpt-3.5-turbo
        prompt=prompt,
        max_tokens=60,
        temperature=0.2
    )
    
    sentiment = response['choices'][0]['text'].strip()
    return sentiment

# Function to Generate Embeddings for FAQ
def generate_embeddings(faq_data):
    embeddings = []
    for faq in faq_data:
        faq_text = faq['question'] + " " + faq['answer']  # Combine question and answer
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=faq_text
        )
        embeddings.append(response['data'][0]['embedding'])
    
    return embeddings

# Function to Build FAISS Index
def build_faq_index(faq_embeddings):
    embeddings_np = np.array(faq_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Using L2 distance for search
    index.add(embeddings_np)  # Add all embeddings to the index
    return index

# Function to Search FAQ Based on Query
def search_faq(query, faq_embeddings, index, faq_data):
    query_embedding = generate_embeddings([{'question': query, 'answer': ''}])[0]
    
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    _, indices = index.search(query_embedding, k=3)  # Top 3 most similar FAQs
    
    relevant_faqs = [faq_data[i] for i in indices[0]]
    return relevant_faqs

# Function to Generate Response
def generate_response(review, sentiment, faq_data, faq_index):
    if sentiment == "negative":
        # Search for relevant FAQs for negative review
        relevant_faqs = search_faq(review, faq_data, faq_index, faq_data)
        
        # Craft a response based on FAQ content
        faq_responses = "\n".join([f"FAQ: {faq['answer']}" for faq in relevant_faqs])
        response = f"Sorry to hear about your experience. Here are some suggestions to help: \n{faq_responses}"
    
    elif sentiment == "positive":
        # Craft a response for positive review
        response = "Thank you for the great review! We're so glad you're enjoying the app. Don't forget to check out our other amazing features!"
    
    return response

# Streamlit App
st.title("Smart App Review Responder")
st.write("Upload a CSV or Excel file with app reviews.")
uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

if uploaded_file:
    # Check the file type
    if uploaded_file.name.endswith('.csv'):
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("### Columns in the uploaded Excel file:")
        st.write(df.columns)  # Display the column names
    # Ensure there is a 'Review' column
    if 'title' not in df.columns:
        st.error("The file must contain a column named 'Review'. Please check your file.")
    else:
        openai_api_key = st.text_input("Enter your OpenAI API Key:")
        
        # Generate embeddings for FAQ data (you can load this from a file)
        faq_data = [
            {'question': 'How do I reset my password?', 'answer': 'You can reset your password by going to the settings page.'},
            {'question': 'How do I change my email?', 'answer': 'You can change your email in the profile section.'}
        ]
        faq_embeddings = generate_embeddings(faq_data)
        faq_index = build_faq_index(faq_embeddings)
        
        # Create a list to store reviews and responses
        review_responses = []
        
        # Process each review
        for review in df['title']:
            sentiment = analyze_sentiment(review)
            response = generate_response(review, sentiment, faq_data, faq_index)
            review_responses.append([review, response])
        
        # Convert the list into a DataFrame
        response_df = pd.DataFrame(review_responses, columns=["Review", "Response"])
        
        # Save the DataFrame to an Excel file
        output_file = "review_responses.xlsx"
        response_df.to_excel(output_file, index=False, engine='openpyxl')
        
        st.write("Responses have been generated. You can download the results below.")
        
        # Provide a download button for the Excel file
        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Responses as Excel",
                data=f,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
