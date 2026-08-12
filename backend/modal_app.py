import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", 
        "uvicorn", 
        "pydantic", 
        "torch", 
        "transformers", 
        "scikit-learn", 
        "xgboost", 
        "spacy", 
        "joblib"
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_dir(".", remote_path="/root/backend")
)

app = modal.App("miro-ai-detection")

@app.function(image=image)
@modal.asgi_app()
def serve():
    import sys
    sys.path.append("/root/backend")
    from app import app as fastapi_app
    return fastapi_app
