import re
import spacy
import requests
from bs4 import BeautifulSoup
from spacy.lang.en.stop_words import STOP_WORDS

import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from heapq import nlargest
from string import punctuation
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
# Tải mô hình ngôn ngữ tiếng Anh từ spaCy
st.set_page_config(layout="wide")
nlp = spacy.load("en_core_web_lg")

with open('./model/s_tokenizer.pkl', 'rb') as f:
    s_tokenizer = pickle.load(f)

# Load the model summary
enc_model = tf.keras.models.load_model('./model/encoder_model.h5')
dec_model = tf.keras.models.load_model('./model/decoder_model.h5')
# Load model translate
tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
model = TFAutoModelForSeq2SeqLM.from_pretrained("../model/tf_model/")

# Biểu thức chính quy để xác định các ký tự đặc biệt nằm giữa hai số
pattern = r'(?<!/d)[^/w/s%](?!/d)'

# ---------------------------------------------------------------------------------------------
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
@st.cache_resource
def extract_summarize(text, per=0.1):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    # tạo từ điển để lưu lại tần số các từ
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    #chuẩn hóa từ bằng cách chia tần suất max
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    # tính điểm = tổng tần suất từ trong câu
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    # xác định số câu và in ra các câu có số điểm từ cao nhất
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    return summary
# ---------------------------------------------------------------------------------------------
@st.cache_resource
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
# ---------------------------------------------------------------------------------------------
@st.cache_resource
def preprocessing(sentence, type=None):
    doc = nlp(sentence)
    lemmatized_text = " ".join([token.lemma_ for token in doc]).lower().strip()
    if type == "gener": 
        # Xóa các ký tự đặc biệt không thuộc trường hợp đã nêu
        lemmatized_text = re.sub(fr'(?<!/d)[^a-zA-Z0-9/s]|[^a-zA-Z0-9/s%](?!/d)|{pattern}', '', lemmatized_text)
        lemmatized_text = re.sub(r'/s+', ' ', lemmatized_text)
    else:
        lemmatized_text = re.sub(r'/s+', ' ', lemmatized_text)
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
    if st.button("Seq2Seq Model"):
        # Trích xuất thông tin từ URL
        extracted_text = extract_text(url, source)
        text = preprocessing(extracted_text, type="gener")
        summary = text_summary(text)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tóm tắt")
            st.write(summary)

        with col2:
            st.subheader("Dịch")
            tokenized = tokenizer([summary], return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                output = tokenizer.decode(out[0], skip_special_tokens=True)
            st.write(output[3:])

    if st.button("Extractive Model"):
        extracted_text = extract_text(url, source)
        text = preprocessing(extracted_text)
        summary = extract_summarize(text)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tóm tắt")
            st.write(summary)

        with col2:
            st.subheader("Dịch")
            tokenized = tokenizer([summary], return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                output = tokenizer.decode(out[0], skip_special_tokens=True)
            st.write(output)
    
# ---------------------------------------------------------------------------------------------
elif choice == "Tóm tắt đoạn văn":
    st.subheader("Summarize Document From Your Text")
    user_text = st.text_area("Paste your text here:", "", key="user_text")
    if st.button("Seq2Seq Model"):
        text = preprocessing(user_text, type="gener")
        summary = text_summary(text)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tóm tắt")
            st.write(summary)

        with col2:
            st.subheader("Dịch")
            tokenized = tokenizer([summary], return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                output = tokenizer.decode(out[0], skip_special_tokens=True)
            st.write(output[3:])

    if st.button("Extractive Model"):
        text = preprocessing(user_text)
        summary = extract_summarize(text)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tóm tắt")
            st.write(summary)

        with col2:
            st.subheader("Dịch")
            tokenized = tokenizer([summary], return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                output = tokenizer.decode(out[0], skip_special_tokens=True)
            st.write(output)