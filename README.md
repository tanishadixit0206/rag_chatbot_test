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
## Running the project using Docker (Recommended Approach)

### If you do not have the Tarball Image:
You will first have to build the docker image and then run it inside a docker container.

For building the image (This process might take a long time to execute depending on your device specifications):
```bash
sudo docker build -t alice_in_wonderland_chatbot .
```

Now you can simply run the image in interactive mode:
```bash
sudo docker run -it alice_in_wonderland_chatbot -q "Your query here" #or you can use --query_text in place of -q
# You may also run it without the query. As soon as the container runs, you will eventually be prompted to enter it.
sudo docker run -it alice_in_wonderland_chatbot
```

### If you have the Tarball Image:

Load the Image:
```bash
docker load < alice_in_wonderland_chatbot.tar
```
Run the image in interactive mode:
```bash
sudo docker run -it alice_in_wonderland_chatbot -q "Your query here" #or you can use --query_text in place of -q
# You may also run it without the query. As soon as the container runs, you will eventually be prompted to enter it.
sudo docker run -it alice_in_wonderland_chatbot
```

## Manual Setup 

### Create a Virtual Environment
Next, we create a virtual environment and activate it

For linux/macOS:
```bash
python3 -m venv venv #or python -m venv venv
source venv/bin/activate
```

For Windows:
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
# You may also run it without the query. As soon as the container runs, you will eventually be prompted to enter it.
sudo docker run -it alice_in_wonderland_chatbot
```

## Conclusion
This project demonstrates a basic RAG-based chatbot using the text from Alice in Wonderland. By leveraging vector-based similarity search and a generative transformer model, it can answer questions from the book with contextually relevant responses.
