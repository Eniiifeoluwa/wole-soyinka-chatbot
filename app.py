import streamlit as st
import torch
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
import json
import torch.nn as nn
nltk.download('punkt')
# Define the ChatModel class
class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Load the model and data
input_size = len(pickle.load(open('all_words.pkl', 'rb')))
hidden_size = 8
output_size = len(pickle.load(open('classes.pkl', 'rb')))

model = ChatModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('chat_model.pth'))
model.eval()

with open('all_words.pkl', 'rb') as f:
    all_words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

def tokenize_and_stem(word):
    return stemmer.stem(word.lower())

def create_bow(pattern, all_words):
    bag_of_words = [0 for _ in range(len(all_words))]
    for word in word_tokenize(pattern):
        stemmed_word = tokenize_and_stem(word)
        if stemmed_word in all_words:
            bag_of_words[all_words.index(stemmed_word)] = 1
    return np.array(bag_of_words)

def predict_class(sentence):
    bow = create_bow(sentence, all_words)
    with torch.no_grad():
        output = model(torch.tensor(bow, dtype=torch.float32).unsqueeze(0))
    _, predicted = torch.max(output, dim=1)
    return classes[predicted.item()]

with open('Dataset.json', 'r') as file:
    data = json.load(file)

responses = {}
for intent in data['book']:
    responses[intent['tag']] = intent['responses']

st.title("Chatbot for 'Death and the King's Horseman'")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'welcomed' not in st.session_state:
    st.session_state.welcomed = False

if not st.session_state.welcomed:
    welcome_prompt = st.text_input("Say 'hello' to start the conversation:", key="welcome_prompt")
    if welcome_prompt.lower() == 'hello':
        welcome_response = "Welcome! I'm here to discuss 'Death and the King's Horseman' by Wole Soyinka. How can I assist you today?"
        st.session_state.history.append({'question': welcome_prompt, 'answer': welcome_response})
        st.write(f"**User😍:** {welcome_prompt}")
        st.write(f"**Chatbot😎:** {welcome_response}")
        st.session_state.welcomed = True
else:
    for entry in st.session_state.history:
        st.write(f"**User😍:** {entry['question']}")
        st.write(f"**Chatbot😎:** {entry['answer']}")

    prompt = st.text_input("Enter your prompt:", key="conversation_prompt")

    if prompt:
        predicted_class = predict_class(prompt)
        response = np.random.choice(responses.get(predicted_class, ["Sorry, I didn't understand that."]))
        st.session_state.history.append({'question': prompt, 'answer': response})
        st.write(f"**User😍:** {prompt}")
        st.write(f"**Chatbot😎:** {response}")
