def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()
def preprocess_text(text):
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text