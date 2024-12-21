from langchain.document_loaders import TextLoader

BOOK_PATH = "./docs/alice_in_wonderland.txt"

def load_book():       
    loader = TextLoader(BOOK_PATH)
    documents = loader.load()
    print('The document has been loaded')

    return documents