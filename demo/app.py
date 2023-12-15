import re
import spacy
import requests
from bs4 import BeautifulSoup

import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# Tải mô hình ngôn ngữ tiếng Anh từ spaCy
st.set_page_config(layout="wide")
nlp = spacy.load("en_core_web_lg")

# load model summary
# Load the tokenizer
with open('./model/s_tokenizer.pkl', 'rb') as f:
    s_tokenizer = pickle.load(f)

# Load the model
enc_model = tf.keras.models.load_model('./model/encoder_model.h5')
dec_model = tf.keras.models.load_model('./model/decoder_model.h5')

# Biểu thức chính quy để xác định các ký tự đặc biệt nằm giữa hai số
pattern = r'(?<!\d)[^\w\s%](?!\d)'

@st.cache_resource
def text_summary(input_text):
    input_seq = s_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=800, padding='post')

    h, c = enc_model.predict(input_seq)
    
    next_token = np.zeros((1, 1))
    next_token[0, 0] = s_tokenizer.word_index['sostok']
    output_seq = ''
    
    stop = False
    count = 0
    
    while not stop:
        if count > 100:
            break
        decoder_out, state_h, state_c = dec_model.predict([next_token]+[h, c])
        token_idx = np.argmax(decoder_out[0, -1, :])
        
        if token_idx == s_tokenizer.word_index['eostok']:
            stop = True
        elif token_idx > 0 and token_idx != s_tokenizer.word_index['sostok']:
            token = s_tokenizer.index_word[token_idx]
            output_seq = output_seq + ' ' + token
        
        next_token = np.zeros((1, 1))
        next_token[0, 0] = token_idx
        h, c = state_h, state_c
        count += 1
    return output_seq.strip()

# ---------------------------------------------------------------------------------------------
def extract_text(url, news_name):
    result = ""
    try:
        response =requests.get(url)
        response.raise_for_status()  # Trả ra ngoại lệ nếu gặp lỗi.
        html_content = response.text # Lấy nội dung HTML phản hồi.
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    if news_name == "CNN": # Nội dung bài báo của CNN được lưu trong các thẻ <p> trong thẻ <main>
        main_content = soup.find('main')
        if main_content:
            paragraphs = main_content.find_all('p')
            for p in paragraphs:
                result += p.get_text()
            return result
        else:
            print("Can't extract text from this news")
            return None
        
    elif news_name == "DailyMail": # Nội dung bài báo của Daily Mail được lưu trong thẻ <div itemprop='articleBody'>
        article_body = soup.find('div', {'itemprop': 'articleBody'})
        if article_body:
            paragraphs = article_body.find_all('p')
            for p in paragraphs:
                result += p.get_text()
            return result
        else:
            print("Can't extract text from this news")
            return None

def preprocessing(sentence):
    doc = nlp(sentence)
    lemmatized_text = " ".join([token.lemma_ for token in doc]).lower().strip()

    # Xóa các ký tự đặc biệt không thuộc trường hợp đã nêu
    lemmatized_text = re.sub(fr'(?<!\d)[^a-zA-Z0-9\s]|[^a-zA-Z0-9\s%](?!\d)|{pattern}', '', lemmatized_text)
    lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text)

    return lemmatized_text

# ---------------------------------------------------------------------------------------------
choice = st.sidebar.selectbox("Chọn chức năng bạn muốn", ["Tóm tắt tin tức", "Tóm tắt đoạn văn"])

if choice == "Tóm tắt tin tức":
    st.subheader("Summarize News From URL")
    # Tạo ô nhập URL và lựa chọn bài báo:
    url = st.text_input("Enter URL:")
    if not url:
        st.warning("Please enter a URL.")
    # Tạo ô nhập trang web muốn lấy thông tin
    source = st.selectbox("Nguồn tin tức:", ["CNN", "DailyMail"])

    if st.button("Extract Text"):
        # Trích xuất thông tin từ URL
        extracted_text = extract_text(url, source)
        text = preprocessing(extracted_text)
        summary = text_summary(text)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tóm tắt")
            st.write(summary)

        with col2:
            st.subheader("Dịch")
            st.write(extracted_text)
            # You can add your summarization logic here
            # Example: st.write(text_summary(preprocessed_text))
        # if extracted_text:
        #     st.header("Extracted Text:")
        #     st.write(text)
        # else:
        #     st.warning("Failed to extract text from the URL.")

    # # Summarize button
    # if st.button("Summarize"):
    #     # Check if URL is entered
    #     if url:
    #         # Summarize news and display result
    #         summary_result = summarize_news(url, source)
    #         if summary_result:
    #             st.header("Summary:")
    #             st.write(summary_result)
    #     else:
    #         st.warning("Please enter a URL before summarizing.")
            
# ---------------------------------------------------------------------------------------------
elif choice == "Summarize Text":
    st.subheader("Summarize Document From Your Text")
    # input_file = st.file_uploader("Upload your document here", type=['pdf'])
    # if input_file is not None:
    #     if st.button("Summarize Document"):
    #         with open("doc_file.pdf", "wb") as f:
    #             f.write(input_file.getbuffer())
    #         col1, col2 = st.columns([1,1])
    #         with col1:
    #             st.info("File uploaded successfully")
    #             extracted_text = extract_text_from_pdf("doc_file.pdf")
    #             st.markdown("**Extracted Text is Below:**")
    #             st.info(extracted_text)
    #         with col2:
    #             st.markdown("**Summary Result**")
    #             text = extract_text_from_pdf("doc_file.pdf")
    #             doc_summary = text_summary(text)
    #             st.success(doc_summary)