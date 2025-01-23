import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST  # Import IST from track_utils
from googletrans import Translator
from deep_translator import GoogleTranslator
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Define the dataset (replace with your actual dataset)
df = pd.DataFrame({
    "text": ["I am so happy today", "I am feeling sad", "This is frustrating"],  # Example text
    "emotion": ["joy", "sadness", "anger"]  # Example emotions
})

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["emotion"], test_size=0.2)

# Map emotions to integers
label_mapping = {label: idx for idx, label in enumerate(df["emotion"].unique())}
train_labels = [label_mapping[label] for label in train_labels]
val_labels = [label_mapping[label] for label in val_labels]

# Convert data into Hugging Face dataset format
train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_data = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_mapping))

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

# Set format for PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert_emotion")
tokenizer.save_pretrained("fine_tuned_bert_emotion")


# Load Model and Vectorizer
pipe_lr2 = joblib.load(open(r"C:\Users\Rohit\Downloads\Text-to-Emotion-main\Text-to-Emotion-main\models\emotion_classifier_pipe_lr2.pkl", "rb"))
vectorizer = joblib.load(open(r"C:\Users\Rohit\Downloads\Text-to-Emotion-main\Text-to-Emotion-main\models\vectorizer.pkl", "rb"))

# Function to translate text to English
# def translate_to_english(text):
#     translator = Translator()
#     translated_text = translator.translate(text, dest='en').text
#     return translated_text
def translate_to_english(text):
    translator = GoogleTranslator(source='auto', target='en')
    translated_text = translator.translate(text)
    return translated_text


# Function to predict emotions
def predict_emotions2(docx):
    translated_text = translate_to_english(docx)
    # text_vec = vectorizer.transform([translated_text])
    results = pipe_lr.predict([translated_text])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba2(docx):
    translated_text = translate_to_english(docx)
    # text_vec = vectorizer.transform([translated_text])
    results = pipe_lr.predict_proba([translated_text])
    return results

# Load Model
pipe_lr = joblib.load(open(r"C:\Users\Rohit\Downloads\Text-to-Emotion-main\Text-to-Emotion-main\models\emotion_classifier_pipe_lr.pkl", "rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Emotion Detection in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions2(raw_text)
            probability = get_prediction_proba2(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")

        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Emotion Detection")

        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")

        st.markdown("##### 2. Confidence Score")

        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")

        st.markdown("##### 3. User-friendly Interface")

        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")

        st.subheader("Applications")

        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)


if __name__ == '__main__':
    main()