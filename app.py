# import streamlit as st
# import io
# import PyPDF2
# import google.generativeai as genai
# from nltk.tokenize import word_tokenize
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# def calculate_similarity(user_message, pdf_content):
#     # Tokenize user message and PDF content
#     user_tokens = word_tokenize(user_message.lower())
#     pdf_tokens = word_tokenize(pdf_content.lower())
#
#     # Combine tokens into sentences
#     user_sentence = ' '.join(user_tokens)
#     pdf_sentence = ' '.join(pdf_tokens)
#
#     # Vectorize the sentences using TF-IDF
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([user_sentence, pdf_sentence])
#
#     # Calculate cosine similarity between user message and PDF content
#     similarity = cosine_similarity(vectors)[0][1]
#     return similarity
#
# import numpy as np
# # Configure GenAI
# genai.configure(api_key="AIzaSyBGr8QJ-5E_IY2DlhKL668swEVq_PCGs80")
#
# def extract_text_from_pdf(file_buffer):
#     text = ""
#     pdf_reader = PyPDF2.PdfReader(file_buffer)
#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         text += page.extract_text()
#     return text
#
#
# def reply(user_message, pdf_content):
#     # Calculate similarity between user message and PDF content
#     similarity = calculate_similarity(user_message, pdf_content)
#     print("Similarity Score:", similarity)  # Print similarity score
#
#     # Define similarity threshold
#     threshold = 0.01
#
#     # If similarity is below threshold, respond with "Sorry, I don't know about that."
#     if similarity < threshold:
#         return "Sorry, I don't know about that."
#     # Set up the model
#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 100,
#     }
#
#     safety_settings = [
#         {
#             "category": "HARM_CATEGORY_HARASSMENT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_HATE_SPEECH",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#     ]
#
#     model = genai.GenerativeModel(model_name="gemini-1.0-pro",
#                                   generation_config=generation_config,
#                                   safety_settings=safety_settings)
#
#     # Start conversation with user's message and PDF content
#     convo = model.start_chat(history=[
#         {
#             "role": "user",
#             "parts": [user_message]
#         },
#         {
#             "role": "model",
#             "parts": [""]
#         },
#     ])
#
#     # Send user's message and PDF content
#     convo.send_message(user_message)
#     convo.send_message(pdf_content)
#
#     # Check if there is a last message in the conversation
#     if convo.last is not None:
#         response = convo.last.text
#     else:
#         response = "Sorry, I don't know about that."
#
#     return response
#
# # Streamlit UI
# st.title("📄 Document Chatbot with GenAI 🤖")
#
# uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
# user_message = st.text_input("💬 Enter your message")
#
# if uploaded_file is not None:
#     # Read the PDF content
#     pdf_content = extract_text_from_pdf(uploaded_file)
#     st.write("PDF Content:")
#     st.write(pdf_content)
#
#     # Get response from the chatbot
#     if st.button("Ask 🎤"):
#         response = reply(user_message, pdf_content)
#         st.write("🤖 Chatbot Response:")
#         st.write(response)

# import streamlit as st
# import io
# import PyPDF2
# import google.generativeai as genai
# from nltk.tokenize import word_tokenize
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # Configure GenAI
# genai.configure(api_key="AIzaSyBGr8QJ-5E_IY2DlhKL668swEVq_PCGs80")
#
# def extract_text_from_pdf(file_buffer):
#     text = ""
#     pdf_reader = PyPDF2.PdfReader(file_buffer)
#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         text += page.extract_text()
#     return text
#
# def calculate_similarity(user_message, pdf_content):
#     # Tokenize user message and PDF content
#     user_tokens = word_tokenize(user_message.lower())
#     pdf_tokens = word_tokenize(pdf_content.lower())
#
#     # Combine tokens into sentences
#     user_sentence = ' '.join(user_tokens)
#     pdf_sentence = ' '.join(pdf_tokens)
#
#     # Vectorize the sentences using TF-IDF
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([user_sentence, pdf_sentence])
#
#     # Calculate cosine similarity between user message and PDF content
#     similarity = cosine_similarity(vectors)[0][1]
#     return similarity
#
# def reply(user_message, pdf_content, chat_with_document=True):
#     if chat_with_document:
#         # Calculate similarity between user message and PDF content
#         similarity = calculate_similarity(user_message, pdf_content)
#
#         # Define similarity threshold
#         threshold = 0.01
#
#         # If similarity is below threshold, respond with "Sorry, I don't know about that."
#         if similarity < threshold:
#             return "Sorry, I don't know about that."
#         # Set up the model
#         generation_config = {
#             "temperature": 0.7,
#             "top_p": 1,
#             "top_k": 1,
#             "max_output_tokens": 100,
#         }
#
#         safety_settings = [
#             {
#                 "category": "HARM_CATEGORY_HARASSMENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_HATE_SPEECH",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#         ]
#
#         model = genai.GenerativeModel(model_name="gemini-1.0-pro",
#                                       generation_config=generation_config,
#                                       safety_settings=safety_settings)
#
#         # Start conversation with user's message and PDF content
#         convo = model.start_chat(history=[
#             {
#                 "role": "user",
#                 "parts": [user_message]
#             },
#             {
#                 "role": "model",
#                 "parts": [""]
#             },
#         ])
#
#         # Send user's message and PDF content
#         convo.send_message(user_message)
#         convo.send_message(pdf_content)
#
#         # Check if there is a last message in the conversation
#         if convo.last is not None:
#             response = convo.last.text
#         else:
#             response = "Sorry, I don't know about that."
#
#         return response
#     else:
#         import google.generativeai as genai
#
#         genai.configure(api_key="AIzaSyBGr8QJ-5E_IY2DlhKL668swEVq_PCGs80")
#
#         # Set up the model
#         generation_config = {
#             "temperature": 0.9,
#             "top_p": 1,
#             "top_k": 1,
#             "max_output_tokens": 2048,
#         }
#
#         safety_settings = [
#             {
#                 "category": "HARM_CATEGORY_HARASSMENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_HATE_SPEECH",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#         ]
#
#         model = genai.GenerativeModel(model_name="gemini-1.0-pro",
#                                       generation_config=generation_config,
#                                       safety_settings=safety_settings)
#
#         convo = model.start_chat(history=[
#             {
#                 "role": "user",
#                 "parts": [user_message]
#             },
#             {
#                 "role": "model",
#                 "parts": [""]
#             },
#         ])
#
#         convo.send_message(user_message)
#
#         return convo.last.text
#
# # Streamlit UI
# st.title("👩‍💻 Chat with Kashish 🤖")
#
# option = st.sidebar.selectbox("Select an option:", ["Kashish Chat", "Open Chat"])
#
# if option == "Kashish Chat":
#     uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
#     user_message = st.text_input("💬 Hi, I'm Kashish! How can I help you today?")
#
#     if uploaded_file is not None:
#         # Read the PDF content
#         pdf_content = extract_text_from_pdf(uploaded_file)
#         st.write("PDF Content:")
#         st.write(pdf_content)
#
#         # Get response from the chatbot
#         if st.button("Ask 🎤"):
#             response = reply(user_message, pdf_content, chat_with_document=True)
#             st.write("🤖 Kashish's Response:")
#             st.write(response)
#
# else:
#     user_message = st.text_input("💬 Hi, I'm Kashish! How can I help you today?")
#
#     if st.button("Ask 🎤"):
#         response = reply(user_message, pdf_content=None, chat_with_document=False)
#         st.write("🤖 Kashish's Response:")
#         st.write(response)


import streamlit as st
import io
import PyPDF2
import google.generativeai as genai
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pytesseract

# Configure Tesseract path (adjust the path based on your installation)
# For Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
# For macOS
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
# For Linux
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configure GenAI
genai.configure(api_key="AIzaSyBGr8QJ-5E_IY2DlhKL668swEVq_PCGs80")

def extract_text_from_pdf(file_buffer):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_buffer)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def calculate_similarity(user_message, pdf_content):
    # Tokenize user message and PDF content
    user_tokens = word_tokenize(user_message.lower())
    pdf_tokens = word_tokenize(pdf_content.lower())

    # Combine tokens into sentences
    user_sentence = ' '.join(user_tokens)
    pdf_sentence = ' '.join(pdf_tokens)

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_sentence, pdf_sentence])

    # Calculate cosine similarity between user message and PDF content
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

def reply(user_message, pdf_content, chat_with_document=True):
    if chat_with_document:
        # Calculate similarity between user message and PDF content
        similarity = calculate_similarity(user_message, pdf_content)

        # Define similarity threshold
        threshold = 0.01

        # If similarity is below threshold, respond with "Sorry, I don't know about that."
        if similarity < threshold:
            return "Sorry, I don't know about that."

        # Set up the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 100,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        # Start conversation with user's message and PDF content
        convo = model.start_chat(history=[
            {
                "role": "user",
                "parts": [user_message]
            },
            {
                "role": "model",
                "parts": [""]
            },
        ])

        # Send user's message and PDF content
        convo.send_message(user_message)
        convo.send_message(pdf_content)

        # Check if there is a last message in the conversation
        if convo.last is not None:
            response = convo.last.text
        else:
            response = "Sorry, I don't know about that."

        return response
    else:
        # Set up the model
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[
            {
                "role": "user",
                "parts": [user_message]
            },
            {
                "role": "model",
                "parts": [""]
            },
        ])

        convo.send_message(user_message)

        return convo.last.text

# Streamlit UI
st.title("👩‍💻 Chat with Kashish 🤖")

option = st.sidebar.selectbox("Select an option:", ["Kashish Chat", "Open Chat"])

if option == "Kashish Chat":
    uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")
    user_message = st.text_input("💬 Hi, I'm Kashish! How can I help you today?")

    if uploaded_file is not None:
        # Read the PDF content
        pdf_content = extract_text_from_pdf(uploaded_file)
        st.write("PDF Content:")
        st.write(pdf_content)

        # Get response from the chatbot
        if st.button("Ask 🎤"):
            response = reply(user_message, pdf_content, chat_with_document=True)
            st.write("🤖 Kashish's Response:")
            st.write(response)

else:
    uploaded_file = st.file_uploader("📂 Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    user_message = st.text_input("💬 Hi, I'm Kashish! How can I help you today?")

    pdf_content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Read the PDF content
            pdf_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            # Read the image content
            image = Image.open(uploaded_file)
            pdf_content = extract_text_from_image(image)
        st.write("Extracted Content:")
        st.write(pdf_content)

    if st.button("Ask 🎤"):
        if pdf_content:
            response = reply(user_message, pdf_content=pdf_content, chat_with_document=True)
        else:
            response = reply(user_message, pdf_content=None, chat_with_document=False)
        st.write("🤖 Kashish's Response:")
        st.write(response)



