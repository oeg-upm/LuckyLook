import streamlit as st
import torch
import joblib
import pandas as pd
from model.bert_gnn import bert_gnn
from model.bert_gnn_PT import bert_gnn_PT
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import pathlib

# Functions
@st.cache_data
def load_model():
    # Prepare model for testing
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        model = bert_gnn_PT("allenai/cs_roberta_base", 768, 469, device=device).to(device)
        checkpoint = torch.load('./saved/models/csRoBERTa_GNN_PT/model_best_csRoBERTa_GNN_PT.pth', map_location=torch.device(device))
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    finally:
        pathlib.PosixPath = posix_backup
    return model

@st.cache_data
def load_labels():
    # prepare labels
    label_encoder = joblib.load('label_encoder.pkl')
    return label_encoder

@st.cache_data
def read_unique_journals(file_path):
    with open(file_path, 'r') as file:
        items = [line.strip() for line in file]
    return items

# Streamlit code
st.title("PubMed Journal Recommender")


st.write("Accuracy: (ACC@1=0.771, ACC@3=0.918, ACC@5=0.951, ACC@10=0.977)")
items = read_unique_journals('unique_journals.txt')

# Sidebar
k = st.sidebar.number_input("Select the desired number of top predictions", min_value=1, max_value=20, value=3)
device = st.sidebar.selectbox('Select a device:', ['cuda', 'cpu'])
selected_journal = st.sidebar.selectbox('Journals:', items)
st.sidebar.write(f"{selected_journal}")

# This will run once and then be cached for future runs
model = load_model()
label_encoder = load_labels()

with st.form("Predict Journal"):
    title = st.text_input("Title", "")
    abstract = st.text_area("Abstract", "")
    keyword = st.text_input("Keyword", "")

    submitted = st.form_submit_button("Submit")
    if submitted:
        with torch.no_grad():
            doc = title + ' ' + abstract + ' ' + (keyword if keyword else ' ')
            tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)
            encoding = tokenizer(doc,truncation=True,padding="max_length",max_length=512)
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            input_ids = encoding['input_ids'].reshape(1, 512)
            mask = encoding['attention_mask'].reshape(1, 512)
            input_ids, mask= input_ids.to(device), mask.to(device)
            output = model(input_ids, mask)
            top_values, top_indices = torch.topk(F.softmax(output, dim=1), k, dim=1)

            top_indices = top_indices.to(device)
            result_labels = label_encoder.inverse_transform(top_indices.numpy().flatten())

            i = 1
            for label, probability in zip(result_labels, top_values.flatten()):
                st.write(f'{i}. {label}, Probability: {probability}')
                i += 1
            