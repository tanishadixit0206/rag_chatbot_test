# Alice in Wonderland Chatbot

## Introduction
This project implements a chatbot based on the text of *Alice in Wonderland* using a Retrieval-Augmented Generation (RAG) workflow. The chatbot processes the book's contents, stores them in a Chroma vector store, and uses a fine-tuned transformer model (BART) to generate responses based on user queries.

### Components Used:
- **Chroma**: A vector database for storing and retrieving chunks of text.
- **SentenceTransformers**: For generating embeddings of text for efficient similarity search.
- **Hugging Face's BART model**: For summarization and generating coherent responses.

The system allows users to ask questions based on the text of *Alice in Wonderland* and will provide a relevant answer derived from the book's content.

## Setup Instructions

### Prerequisites
Before you start, ensure you have the following installed:
- Python 3.7 or higher
- `pip` for package management
- `git` for version control

### Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/baync180705/rag_chatbot.git
cd rag_chatbot
```

### Create a Virtual Environment
Next, we create a virtual environment and activate it

For linux/macOS:
```bash
python3 -m venv venv #or python -m venv venv
source venv/bin/activate
```

For Wondows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
After activating the virtual environment, we install the dependencies

```bash
pip install -r requirements.txt
```

### Run the Project
Now since all the dependencies have been installed, we are good to run the project

```bash
python main.py -q "Your query here" #or you can use --query_text in place of -q
```

### Conclusion
This project demonstrates a basic RAG-based chatbot using the text from Alice in Wonderland. By leveraging vector-based similarity search and a generative transformer model, it can answer questions from the book with contextually relevant responses.
