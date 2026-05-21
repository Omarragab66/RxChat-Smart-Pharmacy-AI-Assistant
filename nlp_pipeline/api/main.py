from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# علشان نقدر نستورد ملفات src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nlp_pipeline import process_query


app = FastAPI(
    title="RxChat Person 1 NLP API",
    description="Arabic Medical NLP Module: Normalization + Intent + Drug Extraction",
    version="1.0.0"
)


class NLPRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "RxChat Person 1 NLP API is running"
    }


@app.post("/nlp")
def nlp_endpoint(request: NLPRequest):
    result = process_query(request.text)
    return result



