# Chatbot for "Death and the King's Horseman"

This project creates an interactive chatbot to discuss Wole Soyinka's play, "Death and the King's Horseman." It uses Streamlit for the user interface, PyTorch for the machine learning model, and NLTK for natural language processing.

## Features

- **Interactive Chat Interface**: Streamlit-based web interface.
- **Machine Learning Model**: Built with PyTorch Question Answering.
- **Text Processing**: Utilizes NLTK for tokenization and stemming.
- **Dynamic Responses**: Responds to user queries with context-specific answers.

## Prerequisites

- Python 3.7 or higher
- Python packages: `streamlit`, `torch`, `nltk`, `numpy`, `pickle`, `json`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eniiifeoluwa/wole-soyinka-chatbot.git
cd wole-soyinka-chatbot
```
## 2. Install Required Packages
```bash
pip install streamlit torch nltk numpy
```
## 3. Download NLTK Data
```bash
import nltk
nltk.download('punkt')
```
## 4. Prepare Required Files
Ensure the following files are in your project directory:

* all_words.pkl (File with tokenized and stemmed words)
* classes.pkl (File with class labels)
* chat_model.pth (Trained PyTorch model file)
* Dataset.json (JSON file with intents and responses data)

## Running the Application

### 1. Start the Streamlit App

Open your terminal or command prompt, navigate to your project directory, and run:

```bash
streamlit run app.py
```
### 2. Interact with the Chatbot

* Once the Streamlit server starts, it will provide a URL in the terminal. Open this URL in your web browser.
* Type hello to start the conversation.
HEAD
* Ask questions related to "Death and the King's Horseman."
* Ask questions related to "Death and the King's Horseman."

