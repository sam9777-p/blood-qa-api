from flask import Flask, request, jsonify
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
import torch

app = Flask(__name__)

# Load model and tokenizer
model_name = "impira/layoutlm-document-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LayoutLMForQuestionAnswering.from_pretrained(model_name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data["question"]
    context = data["context"]  # This should be the document text
    words = data.get("words", [])
    boxes = data.get("boxes", [])
    
    encoding = tokenizer(question, context, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1

    answer_tokens = encoding["input_ids"][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
