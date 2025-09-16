from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib

model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

app = FastAPI()

# Habilitar CORS para desarrollo (ajusta orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def map_label_and_confidence(text: str):
    """Helper: vectoriza, predice y devuelve (label_es, confidence)."""
    X = vectorizer.transform([text.lower()])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    try:
        pred_index = list(model.classes_).index(pred)
        prob = float(probs[pred_index])
    except Exception:
        prob = float(probs.max())

    # mapear la etiqueta a español (robusto para strings o 0/1)
    if isinstance(pred, (int, float)):
        label = "positivo" if pred == 1 else "negativo"
    else:
        p = str(pred).lower()
        if any(k in p for k in ("pos", "positive", "positivo")):
            label = "positivo"
        elif any(k in p for k in ("neg", "negative", "negativo")):
            label = "negativo"
        else:
            label = str(pred)

    return label, round(prob, 4)


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    label, prob = map_label_and_confidence(text)
    return {"input": text, "prediction": label, "confidence": prob}


@app.get("/predict_get")
async def predict_get(text: str = Query("", description="Texto a clasificar")):
    """Endpoint de conveniencia: permite llamar desde la barra de direcciones o desde fetch en el navegador.
    Uso: /predict_get?text=hola%20mundo
    """
    if not text:
        return {"error": "Proporciona el parámetro 'text' en la querystring."}
    label, prob = map_label_and_confidence(text)
    return {"input": text, "prediction": label, "confidence": prob}