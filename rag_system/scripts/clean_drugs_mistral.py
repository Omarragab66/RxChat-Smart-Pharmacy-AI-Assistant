"""
Drug Data Cleaner using Ollama Mistral
======================================
يقرأ ملف CSV فيه بيانات أدوية
ويبعت كل دواء لـ Mistral عشان ينظفه ويحسنه
ويحفظ النتيجة في ملف CSV جديد
"""

import pandas as pd
import requests
import json
import time
import os
from pathlib import Path
from datetime import datetime

# =====================
# الإعدادات
# =====================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "core_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_drugs_core.csv"

# عدد الثواني بين كل طلب
DELAY_BETWEEN_REQUESTS = 1

# لو عاوز تجرب على عدد محدود الأول (اعمل None لو عاوز الكل)
TEST_LIMIT = None  # مثال: 5 للتجربة، None للكل

# Timeout بالثواني - رفعناه لـ 5 دقايق
REQUEST_TIMEOUT = 300

# =====================
# الـ Prompt - مختصر عشان Mistral يشتغل أسرع
# =====================
SYSTEM_PROMPT = """نظّف بيانات الدواء التالية وحسّن صياغتها:
- احذف التكرار واحتفظ بكل المعلومات الطبية المهمة
- صحّح الأخطاء اللغوية (عربي وإنجليزي)
- لا تضف معلومات جديدة غير موجودة في النص
- اكتب النتيجة منظمة وواضحة بدون تعليقات منك

بيانات الدواء:"""


def build_drug_text(row):
    """بيبني نص الدواء من صفوف الـ CSV"""
    parts = []

    parts.append(f"الدواء: {row.get('drug_name', '')} ({row.get('generic_name', '')})")
    parts.append(f"الفئة الدوائية: {row.get('drug_class', '')}")
    parts.append(f"الشكل الدوائي: {row.get('form', '')} | طريقة الاستخدام: {row.get('route', '')}")

    if pd.notna(row.get('indications')) and str(row.get('indications')).strip():
        parts.append(f"\nالاستخدامات:\n{row['indications']}")

    if pd.notna(row.get('dosage')) and str(row.get('dosage')).strip():
        parts.append(f"\nالجرعة:\n{row['dosage']}")

    if pd.notna(row.get('mechanism')) and str(row.get('mechanism')).strip():
        parts.append(f"\nآلية العمل:\n{row['mechanism']}")

    if pd.notna(row.get('warnings')) and str(row.get('warnings')).strip():
        parts.append(f"\nالتحذيرات:\n{row['warnings']}")

    if pd.notna(row.get('contraindications')) and str(row.get('contraindications')).strip():
        parts.append(f"\nموانع الاستخدام:\n{row['contraindications']}")

    if pd.notna(row.get('drug_interactions')) and str(row.get('drug_interactions')).strip():
        parts.append(f"\nالتفاعلات الدوائية:\n{row['drug_interactions']}")

    if pd.notna(row.get('adverse_reactions')) and str(row.get('adverse_reactions')).strip():
        parts.append(f"\nالآثار الجانبية:\n{row['adverse_reactions']}")

    if pd.notna(row.get('pregnancy_info')) and str(row.get('pregnancy_info')).strip():
        parts.append(f"\nالحمل والرضاعة:\n{row['pregnancy_info']}")

    if pd.notna(row.get('pediatric_info')) and str(row.get('pediatric_info')).strip():
        parts.append(f"\nالأطفال:\n{row['pediatric_info']}")

    if pd.notna(row.get('overdose_info')) and str(row.get('overdose_info')).strip():
        parts.append(f"\nالجرعة الزائدة:\n{row['overdose_info']}")

    if pd.notna(row.get('storage_info')) and str(row.get('storage_info')).strip():
        parts.append(f"\nالتخزين:\n{row['storage_info']}")

    return "\n".join(parts)


def clean_with_mistral(drug_text):
    """بيبعت النص لـ Mistral ويرجع النص المنظف - بيستخدم streaming"""

    prompt = f"{SYSTEM_PROMPT}\n{drug_text}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,   # streaming عشان ما يـ timeout ش
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 1024,
            "num_ctx": 4096,
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=True
        )
        response.raise_for_status()

        # اجمع الـ tokens واحدة واحدة
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    full_response += chunk.get("response", "")
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        return full_response.strip() if full_response else None
    except requests.exceptions.ConnectionError:
        print("❌ خطأ: Ollama مش شغال! تأكد إنك شغلت: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("⏰ خطأ: الطلب أخد وقت طويل جداً")
        return None
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        return None


def check_ollama_connection():
    """بيتأكد إن Ollama شغال"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m['name'] for m in resp.json().get('models', [])]
        print(f"✅ Ollama شغال | الموديلات المتاحة: {models}")

        if not any('mistral' in m for m in models):
            print(f"⚠️  تحذير: mistral مش موجود! شغّل: ollama pull mistral")
            return False
        return True

    except Exception:
        print("❌ Ollama مش شغال! شغّل: ollama serve")
        return False


def main():
    print("=" * 60)
    print("🔬 Drug Data Cleaner - Mistral via Ollama")
    print("=" * 60)

    # تأكد إن Ollama شغال
    if not check_ollama_connection():
        return

    # قرا الـ CSV
    print(f"\n📂 بيقرأ الملف: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ الملف مش موجود: {INPUT_FILE}")
        print("تأكد إن الملف في نفس المجلد مع السكريبت")
        return

    total = len(df)
    if TEST_LIMIT:
        df = df.head(TEST_LIMIT)
        print(f"🧪 وضع التجربة: هيشتغل على أول {TEST_LIMIT} أدوية بس")

    print(f"📊 إجمالي الأدوية: {len(df)} من {total}")

    # عمود جديد للنتيجة المنظفة
    df['cleaned_rag_text'] = ""
    df['cleaning_status'] = ""

    # ابدأ التنظيف
    print(f"\n🚀 بدأ التنظيف...\n")
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        drug_name = row.get('drug_name', f'دواء {idx}')
        print(f"[{idx+1}/{len(df)}] 💊 {drug_name} ... ", end="", flush=True)

        # ابني نص الدواء
        drug_text = build_drug_text(row)

        # بعته لـ Mistral
        start_time = time.time()
        cleaned_text = clean_with_mistral(drug_text)
        elapsed = time.time() - start_time

        if cleaned_text:
            df.at[idx, 'cleaned_rag_text'] = cleaned_text
            df.at[idx, 'cleaning_status'] = 'success'
            success_count += 1
            print(f"✅ ({elapsed:.1f}s)")
        else:
            df.at[idx, 'cleaned_rag_text'] = row.get('rag_text', '')  # احتفظ بالأصلي
            df.at[idx, 'cleaning_status'] = 'failed'
            fail_count += 1
            print(f"❌ فشل - احتفظ بالأصلي")

        # استنى شوية بين الطلبات
        if idx < len(df) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

        # احفظ كل 10 أدوية عشان ما تخسرش شغلك
        if (idx + 1) % 10 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"   💾 حفظ مؤقت: {OUTPUT_FILE}")

    # احفظ النهائي
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # ملخص
    print("\n" + "=" * 60)
    print("📋 ملخص النتائج:")
    print(f"   ✅ نجح: {success_count} دواء")
    print(f"   ❌ فشل: {fail_count} دواء")
    print(f"   💾 الملف المحفوظ: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()




