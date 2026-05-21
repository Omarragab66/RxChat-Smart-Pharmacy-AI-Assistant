# ====================================================
# Flask API for RxChat Safety Classification Model
# ====================================================

import os
from pathlib import Path
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flaYOUR_API_KEY_HERE import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------
# Config
# ----------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rxchat_safety_model_final_fixed"
MAX_LEN = 128

id2label = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

print("⏳ Loading Safety Model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

model = model.to(device)
model.eval()

print("✅ Model loaded successfully")
print("Device:", device)

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------

def predict_safety(text: str):
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    pred_id = torch.argmax(probs).item()
    riYOUR_API_KEY_HERE = id2label[pred_id]
    confidence = probs[pred_id].item()

    return {
        "riYOUR_API_KEY_HERE": riYOUR_API_KEY_HERE,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Low": round(probs[0].item(), 4),
            "Medium": round(probs[1].item(), 4),
            "High": round(probs[2].item(), 4)
        }
    }

# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RxChat Safety Model API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": device
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        # لو فتحت من المتصفح عادي GET
        if request.method == "GET":
            text = request.args.get("text", "")

            # لو مفيش text اعرض صفحة بسيطة
            if text.strip() == "":
                return """
                <!DOCTYPE html>
                <html lang="ar" dir="rtl">
                <head>
                    <meta charset="UTF-8">
                    <title>RxChat Safety Classifier</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            background: #f4f7f8;
                            padding: 40px;
                            direction: rtl;
                        }
                        .container {
                            max-width: 700px;
                            margin: auto;
                            background: white;
                            padding: 25px;
                            border-radius: 12px;
                            box-shadow: 0 0 15px rgba(0,0,0,0.1);
                        }
                        textarea {
                            width: 100%;
                            height: 130px;
                            font-size: 18px;
                            padding: 10px;
                            border-radius: 8px;
                            border: 1px solid #ccc;
                        }
                        button {
                            margin-top: 15px;
                            background: #0d6efd;
                            color: white;
                            border: none;
                            padding: 12px 25px;
                            border-radius: 8px;
                            font-size: 18px;
                            cursor: pointer;
                        }
                        button:hover {
                            background: #084298;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>💊 RxChat Safety Classifier</h2>
                        <p>اكتب سؤال المريض واضغط Predict</p>

                        <form method="GET" action="/predict">
                            <textarea name="text" placeholder="مثال: أنا عندي قصور كلوي شديد ومحتاج آخد فولتارين لوجع الظهر"></textarea>
                            <br>
                            <button type="submit">Predict</button>
                        </form>
                    </div>
                </body>
                </html>
                """

            result = predict_safety(text)

            return f"""
            <!DOCTYPE html>
            <html lang="ar" dir="rtl">
            <head>
                <meta charset="UTF-8">
                <title>Prediction Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f7f8;
                        padding: 40px;
                        direction: rtl;
                    }}
                    .container {{
                        max-width: 700px;
                        margin: auto;
                        background: white;
                        padding: 25px;
                        border-radius: 12px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.1);
                    }}
                    .risk {{
                        font-size: 28px;
                        font-weight: bold;
                        color: #0d6efd;
                    }}
                    pre {{
                        background: #eee;
                        padding: 15px;
                        border-radius: 8px;
                        direction: ltr;
                        text-align: left;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 20px;
                        text-decoration: none;
                        color: white;
                        background: #0d6efd;
                        padding: 10px 20px;
                        border-radius: 8px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>💊 RxChat Safety Prediction</h2>

                    <p><b>النص:</b></p>
                    <p>{text}</p>

                    <p><b>التصنيف:</b></p>
                    <p class="risk">{result["riYOUR_API_KEY_HERE"]}</p>

                    <p><b>Confidence:</b> {result["confidence"]}</p>

                    <p><b>Probabilities:</b></p>
                    <pre>{result["probabilities"]}</pre>

                    <a href="/predict">جرب سؤال تاني</a>
                </div>
            </body>
            </html>
            """

        # لو POST من Postman أو أي frontend
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Invalid JSON body"
            }), 400

        text = data.get("text", "")
        result = predict_safety(text)

        return jsonify({
            "input_text": text,
            "prediction": result
        })

    except ValueError as e:
        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500
# ----------------------------------------------------
# Run Server
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )




