from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from detector import Detector

app = FastAPI(title="AI Admissions Essay Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Detector...")
# This might take a few seconds on startup
detector = None

@app.on_event("startup")
def load_detector():
    global detector
    try:
        detector = Detector()
    except Exception as e:
        print(f"Error loading detector: {e}")

class DetectRequest(BaseModel):
    text: str
    
@app.post("/detect")
def detect_essay(request: DetectRequest):
    if not request.text or len(request.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text is too short to reliably analyze.")
        
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector is not fully loaded yet.")
        
    try:
        import explainer
        result = detector.detect(request.text)
        if "feature_contributions" in result and result["feature_contributions"]:
            result["plain_english_explanation"] = explainer.explain_verdict(
                probability=result["ai_probability"],
                feature_contributions=result["feature_contributions"]
            )
        else:
            result["plain_english_explanation"] = "Mathematical weights are not available until the logistic regression model finishes training."
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
