from flask import Flask, request, jsonify
from transformers import LayoutLMTokenizer, LayoutLMForQuestionAnswering
from PIL import Image
import pytesseract
import torch
import os

app = Flask(__name__)

model_name = "impira/layoutlm-document-qa"
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
model = LayoutLMForQuestionAnswering.from_pretrained(model_name)

@app.route('/')
def index():
    return "Impira LayoutLM Document QA API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    question = request.form.get('question', 'What is this document about?')
    image = Image.open(file.stream).convert("RGB")

    text = pytesseract.image_to_string(image)
    if not text.strip():
        return jsonify({"error": "No text found in the image"}), 400

    inputs = tokenizer(question, text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    return jsonify({
        "question": question,
        "answer": answer.strip()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
