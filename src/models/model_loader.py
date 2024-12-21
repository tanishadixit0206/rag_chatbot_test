from sentence_transformers import SentenceTransformer
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

def load_summarizer(model_name):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer
