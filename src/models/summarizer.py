from transformers import BartForConditionalGeneration, BartTokenizer
from .model_loader import load_summarizer

model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def generate_result(prompt): 
    model, tokenizer = load_summarizer(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, min_length=25)   
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace("  ","")