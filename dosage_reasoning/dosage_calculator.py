"""
💊 RxChat — Person 3: Personalized Dosage Reasoning
استخدام DeepSeek-R1 (via Groq) لحساب الجرعة الشخصية
"""
import re
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# 1. Patient Profile Schema
# ─────────────────────────────────────────────

class PatientProfile(BaseModel):
    age: int
    weight_kg: float
    gender: str                          # "ذكر" | "أنثى"
    chronic_conditions: list[str] = []
    kidney_function: str = "normal"      # normal | mild_impairment | moderate_impairment | severe_impairment
    liver_function: str = "normal"       # normal | mild_impairment | moderate_impairment | severe_impairment


class DosageRequest(BaseModel):
    drug_name: str
    patient_profile: PatientProfile


class DosageResponse(BaseModel):
    drug_name: str
    recommended_dose: str
    reasoning: str
    chain_of_thought: list[str]
    safety_flag: bool
    doctor_referral: bool
    dose_adjustment_reason: str
    riYOUR_API_KEY_HERE: str                   # LOW | MEDIUM | HIGH  (من الـ rule-based classifier)


# ─────────────────────────────────────────────
# 2. Chain-of-Thought Prompt Builder
# ─────────────────────────────────────────────

def load_few_shot_examples(json_path: str, n: int = 5) -> list[dict]:
    """تحميل أمثلة عشوائية من الداتا للـ few-shot prompting"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return random.sample(data, min(n, len(data)))


def build_few_shot_block(examples: list[dict]) -> str:
    """تحويل الأمثلة لنص few-shot"""
    blocks = []
    for ex in examples:
        pp = ex["patient_profile"]
        cot = "\n".join(ex["chain_of_thought"])
        block = f"""
=== مثال ===
المريض: عمر {pp['age']} سنة، وزن {pp['weight_kg']} كجم، جنس: {pp['gender']}
الأمراض المزمنة: {', '.join(pp['chronic_conditions']) if pp['chronic_conditions'] else 'لا يوجد'}
وظيفة الكلى: {pp['kidney_function']} | وظيفة الكبد: {pp['liver_function']}
الدواء: {ex['drug_name']}

التحليل خطوة بخطوة:
{cot}

الجرعة الموصى بها: {ex['recommended_dose'] if ex['recommended_dose'] else 'يجب تحويل للطبيب'}
سبب التعديل: {ex['dose_adjustment_reason']}
تحويل للطبيب: {'نعم ⚠️' if ex['doctor_referral'] else 'لا'}
"""
        blocks.append(block)
    return "\n".join(blocks)


def build_deepseek_prompt(drug_name: str, patient: PatientProfile, few_shot_examples: list[dict]) -> str:
    """بناء الـ prompt الكامل لـ DeepSeek-R1"""
    
    few_shot_block = build_few_shot_block(few_shot_examples)
    conditions_str = ", ".join(patient.chronic_conditions) if patient.chronic_conditions else "لا يوجد"

    system_prompt = """أنت نظام ذكاء اصطناعي متخصص في حساب جرعات الأدوية الشخصية.
مهمتك: تحليل حالة المريض وتقديم الجرعة المناسبة بناءً على إرشادات OpenFDA الرسمية.

قواعد مهمة:
- لا تكتب أي محتوى داخل <think>.
- اكتب فقط التحليل الظاهر المختصر في 3-5 خطوات.
- راعِ العمر والوزن والأمراض المزمنة ووظائف الكلى والكبد
- إذا كانت الحالة خطيرة، قل "يجب تحويل للطبيب" ولا تعطِ جرعة محددة
- ردّك يجب أن يكون بالعربية الفصحى

أمثلة:
""" + few_shot_block

    user_prompt = f"""
=== طلب جديد ===
الدواء: {drug_name}
المريض:
  - العمر: {patient.age} سنة
  - الوزن: {patient.weight_kg} كجم
  - الجنس: {patient.gender}
  - الأمراض المزمنة: {conditions_str}
  - وظيفة الكلى: {patient.kidney_function}
  - وظيفة الكبد: {patient.liver_function}

قدّم تحليلك خطوة بخطوة ثم اعطِ الجرعة الموصى بها.
"""

    return system_prompt, user_prompt


# ─────────────────────────────────────────────
# 3. DeepSeek-R1 Caller (via Groq)
# ─────────────────────────────────────────────

def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    """استدعاء DeepSeek-R1 عبر Groq API"""
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to call the dosage reasoning model.")

    client = Groq(api_key=groq_api_key)
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,       # منخفض للحصول على نتائج موثوقة
        max_tokens=1024,
    )
    
    return response.choices[0].message.content

def clean_llm_response(text: str) -> str:
    """
    Remove hidden thinking blocks from LLM response.
    """
    if not text:
        return ""

    text = re.sub(r"<tool_call>.*?<tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
def parse_deepseek_response(raw_response: str) -> dict:
    """
    استخراج المعلومات المهيكلة من رد الموديل
    مع حذف <think> وتصحيح تحويل للطبيب: لا
    """

    clean_response = clean_llm_response(raw_response)

    clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL | re.IGNORECASE)
    clean_response = clean_response.strip()

    lines = clean_response.split("\n")
    chain_of_thought = []
    recommended_dose = ""
    dose_adjustment_reason = ""
    doctor_referral = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # استخراج خطوات التحليل الظاهرة فقط
        if line.startswith("الخطوة") or line.startswith("خطوة"):
            chain_of_thought.append(line)

        # استخراج الجرعة
        if "الجرعة الموصى بها" in line or "الجرعة:" in line:
            recommended_dose = line.split(":", 1)[-1].strip()

        # استخراج سبب التعديل
        if "سبب" in line and "تعديل" in line:
            dose_adjustment_reason = line.split(":", 1)[-1].strip()

        # استخراج قرار تحويل الطبيب بشكل صحيح
        if "تحويل للطبيب" in line:
            referral_value = line.split(":", 1)[-1].strip() if ":" in line else line

            if "لا" in referral_value or "no" in referral_value.lower():
                doctor_referral = False
            elif "نعم" in referral_value or "yes" in referral_value.lower():
                doctor_referral = True

        # عبارات خطيرة واضحة
        if "يجب مراجعة الطبيب" in line or "استشارة الطبيب" in line or "يجب تحويل" in line:
            if "لا" not in line:
                doctor_referral = True

    # لو الموديل ماطلعش خطوات واضحة
    if not chain_of_thought:
        chain_of_thought = [l.strip() for l in lines if len(l.strip()) > 20][:5]

    # هنا safety_flag معناها: هل في خطر يستدعي دكتور؟
    safety_flag = doctor_referral or not recommended_dose

    return {
        "chain_of_thought": chain_of_thought,
        "recommended_dose": recommended_dose,
        "dose_adjustment_reason": dose_adjustment_reason,
        "safety_flag": safety_flag,
        "doctor_referral": doctor_referral,
        "raw_reasoning": clean_response
    }
# ─────────────────────────────────────────────
# 4. Rule-Based Safety Classifier (scikit-learn)
# هذا هو الـ DL component الإضافي المطلوب للـ presentation
# ─────────────────────────────────────────────

def build_features(record: dict) -> list:
    """تحويل Patient Profile لـ features رقمية"""
    pp = record["patient_profile"]
    
    kidney_map = {"normal": 0, "mild_impairment": 1, "moderate_impairment": 2, "severe_impairment": 3}
    liver_map  = {"normal": 0, "mild_impairment": 1, "moderate_impairment": 2, "severe_impairment": 3}

    features = [
        pp["age"],
        pp["weight_kg"],
        1 if pp["gender"] == "ذكر" else 0,
        len(pp["chronic_conditions"]),                          # عدد الأمراض المزمنة
        kidney_map.get(pp["kidney_function"], 0),
        liver_map.get(pp["liver_function"], 0),
        1 if any("سكري" in c for c in pp["chronic_conditions"]) else 0,
        1 if any("قلب" in c or "ضغط" in c for c in pp["chronic_conditions"]) else 0,
        1 if any("كبد" in c for c in pp["chronic_conditions"]) else 0,
        1 if any("كلى" in c or "كلوي" in c for c in pp["chronic_conditions"]) else 0,
    ]
    return features


def get_riYOUR_API_KEY_HERE(record: dict) -> str:
    """تصنيف مستوى الخطورة بناءً على Rules"""
    pp = record["patient_profile"]
    kidney_map = {"normal": 0, "mild_impairment": 1, "moderate_impairment": 2, "severe_impairment": 3}
    liver_map  = {"normal": 0, "mild_impairment": 1, "moderate_impairment": 2, "severe_impairment": 3}
    
    kidney_score = kidney_map.get(pp["kidney_function"], 0)
    liver_score  = liver_map.get(pp["liver_function"], 0)
    conditions   = len(pp["chronic_conditions"])
    age          = pp["age"]

    if kidney_score >= 3 or liver_score >= 3 or conditions >= 3:
        return "HIGH"
    elif kidney_score >= 1 or liver_score >= 1 or conditions >= 1 or age >= 65 or age <= 12:
        return "MEDIUM"
    else:
        return "LOW"


def train_safety_classifier(json_path: str, model_save_path: str = "safety_classifier.pkl"):
    """تدريب الـ Random Forest Classifier على البيانات"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = [build_features(r) for r in data]
    y = [get_riYOUR_API_KEY_HERE(r) for r in data]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== Safety Classifier Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    return clf, le


def load_safety_classifier(model_path: str = "safety_classifier.pkl"):
    """تحميل الـ classifier من الملف"""
    obj = joblib.load(model_path)
    return obj["model"], obj["label_encoder"]


def predict_risk(patient_profile: PatientProfile, clf, le) -> str:
    """توقع مستوى الخطورة لمريض جديد"""
    record = {"patient_profile": patient_profile.dict()}
    features = [build_features(record)]
    pred = clf.predict(features)
    return le.inverse_transform(pred)[0]


# ─────────────────────────────────────────────
# 5. Main Dosage Engine
# ─────────────────────────────────────────────

class DosageEngine:
    def __init__(self, data_path: str, classifier_path: str = "safety_classifier.pkl"):
        self.data_path = data_path
        self.classifier_path = classifier_path
        
        # تحميل أو تدريب الـ classifier
        if os.path.exists(classifier_path):
            self.clf, self.le = load_safety_classifier(classifier_path)
            print("✅ Loaded existing safety classifier")
        else:
            print("🔧 Training safety classifier...")
            self.clf, self.le = train_safety_classifier(data_path, classifier_path)

    def calculate_dose(self, drug_name: str, patient: PatientProfile) -> DosageResponse:
        """الدالة الرئيسية لحساب الجرعة"""
        
        # 1. توقع مستوى الخطورة بالـ rule-based classifier
        riYOUR_API_KEY_HERE = predict_risk(patient, self.clf, self.le)
        
        # 2. تحميل أمثلة few-shot من الداتا
        few_shot_examples = load_few_shot_examples(self.data_path, n=3)
        
        # 3. بناء الـ prompt
        system_prompt, user_prompt = build_deepseek_prompt(drug_name, patient, few_shot_examples)
        
        # 4. استدعاء DeepSeek-R1
        raw_response = call_deepseek(system_prompt, user_prompt)
        
        # 5. تحليل الرد
        parsed = parse_deepseek_response(raw_response)
        
        # 6. دمج مع riYOUR_API_KEY_HERE
        if riYOUR_API_KEY_HERE == "HIGH":
            parsed["safety_flag"] = True
            parsed["doctor_referral"] = True

        return DosageResponse(
            drug_name=drug_name,
            recommended_dose=parsed["recommended_dose"],
            reasoning=parsed["raw_reasoning"],
            chain_of_thought=parsed["chain_of_thought"],
            safety_flag=parsed["safety_flag"],
            doctor_referral=parsed["doctor_referral"],
            dose_adjustment_reason=parsed["dose_adjustment_reason"],
            riYOUR_API_KEY_HERE=riYOUR_API_KEY_HERE
        )


# ─────────────────────────────────────────────
# 6. FastAPI Endpoint
# ─────────────────────────────────────────────

app = FastAPI(title="RxChat — Dosage Calculator", version="1.0")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("DOSAGE_DATA_PATH", BASE_DIR / "data" / "deepseek_examples_full.json"))
CLASSIFIER_PATH = Path(os.getenv("DOSAGE_CLASSIFIER_PATH", BASE_DIR / "models" / "safety_classifier.pkl"))
engine = None  # يتم تهيئته عند بدء التشغيل


@app.on_event("startup")
async def startup():
    global engine
    engine = DosageEngine(data_path=str(DATA_PATH), classifier_path=str(CLASSIFIER_PATH))
    print("✅ Dosage Engine initialized")


@app.post("/dosage", response_model=DosageResponse)
async def calculate_dosage(request: DosageRequest):
    """
    POST /dosage
    Input:  drug_name + patient_profile
    Output: recommended_dose + reasoning + chain_of_thought + safety_flag + doctor_referral
    """
    result = engine.calculate_dose(
        drug_name=request.drug_name,
        patient=request.patient_profile
    )
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "module": "Person 3 — Personalized Dosage"}


# ─────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # تدريب الـ classifier أول مرة
    if not CLASSIFIER_PATH.exists():
        train_safety_classifier(str(DATA_PATH), str(CLASSIFIER_PATH))

    # تشغيل الـ API
    uvicorn.run("dosage_calculator:app", host="0.0.0.0", port=8003, reload=True)




