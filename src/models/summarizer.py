from .model_loader import load_summarizer
from language_tool_python import LanguageTool

model_name = "google/flan-t5-xl"


def generate_result(prompt): 
    model, tokenizer = load_summarizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
    inputs['input_ids'], 
    max_length=300, 
    min_length=50, 
    num_beams=5, 
    temperature=0.7,  
    top_k=50,
    top_p=0.9,
    early_stopping=True,
    do_sample=True
)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not response.endswith(('.', '!', '?')):
        response += '.'
    response = response[0].capitalize() + response[1:]
    
    tool = LanguageTool('en-US')
    corrected_response = tool.correct(response)
    return(corrected_response)
