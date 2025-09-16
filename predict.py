import sys
import joblib

model = joblib.load('models/model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

def predict_text(text):
    clean = text.lower()
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    # encontrar índice de la clase predicha para tomar su probabilidad
    try:
        pred_index = list(model.classes_).index(pred)
        prob = float(probs[pred_index])
    except Exception:
        # fallback: usar la probabilidad máxima si algo raro pasa
        prob = float(probs.max())

    # mapear la etiqueta predicha a español de forma robusta
    if isinstance(pred, (int, float)):
        label = "positivo" if pred == 1 else "negativo"
    else:
        p = str(pred).lower()
        if any(k in p for k in ("pos", "positive", "positivo")):
            label = "positivo"
        elif any(k in p for k in ("neg", "negative", "negativo")):
            label = "negativo"
        else:
            # si no se puede mapear, devolver la etiqueta tal cual
            label = str(pred)

    return {
        "input": text,
        "prediction": label,
        "confidence": round(prob, 5)
    }

if __name__ == "__main__":
    texto = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else " "
    print(predict_text(texto))