from flask import Flask, request, jsonify
from transformers import LayoutLMv2Processor, LayoutLMv2ForQuestionAnswering
from PIL import Image
import pytesseract
import torch
import os

app = Flask(__name__)

model_name = "microsoft/layoutlmv2-base-uncased"
processor = LayoutLMv2Processor.from_pretrained(model_name)
model = LayoutLMv2ForQuestionAnswering.from_pretrained(model_name)

@app.route('/')
def index():
    return "LayoutLMv2 OCR QA service is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    question = request.form.get('question', "Is the person eligible to donate blood?")
    image = Image.open(file.stream).convert("RGB")

    words_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []

    for i in range(len(words_data["text"])):
        if int(words_data["conf"][i]) > 60:
            word = words_data["text"][i]
            if word.strip():
                words.append(word)
                x, y, w, h = (words_data["left"][i], words_data["top"][i],
                              words_data["width"][i], words_data["height"][i])
                box = [x, y, x + w, y + h]
                boxes.append(box)

    if not words:
        return jsonify({"error": "No readable text found"}), 400

    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    bbox = encoding["bbox"]
    token_type_ids = encoding["token_type_ids"]
    image_tensor = encoding["image"]

    question_encoding = processor.tokenizer(question, truncation=True, padding="max_length", return_tensors="pt")
    input_ids[:, :question_encoding["input_ids"].size(1)] = question_encoding["input_ids"]
    attention_mask[:, :question_encoding["attention_mask"].size(1)] = question_encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        image=image_tensor)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start = torch.argmax(start_logits, dim=1).item()
    end = torch.argmax(end_logits, dim=1).item()

    answer_tokens = input_ids[0][start:end+1]
    answer = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return jsonify({
        "question": question,
        "answer": answer.strip()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
