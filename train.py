import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def find_column(df, candidates):
	"""Return the first matching column name from candidates (case-insensitive) or None."""
	cols_lower = {c.lower(): c for c in df.columns}
	for cand in candidates:
		if cand.lower() in cols_lower:
			return cols_lower[cand.lower()]
	return None


def main():
	# Cargar el dataset; soporta CSV con o sin encabezado
	try:
		df = pd.read_csv('data/dataset.csv', header=0)
	except pd.errors.EmptyDataError:
		print("El archivo 'data/dataset.csv' está vacío o no existe. Comprueba la ruta.")
		return
	except Exception as e:
		print(f"Error leyendo 'data/dataset.csv': {e}")
		return

	# Verificar que el dataset tiene columnas de texto y etiqueta (intenta nombres comunes)
	text_col = find_column(df, ['text', 'tweet', 'message', 'content'])
	label_col = find_column(df, ['label', 'sentiment', 'class'])

	# Si no se encontraron, intentar leer sin encabezado y detectar columnas automáticamente
	if text_col is None or label_col is None:
		# Releer sin encabezado (header=None) para manejar CSVs sin cabecera
		try:
			df_no_header = pd.read_csv('data/dataset.csv', header=None)
		except Exception:
			print("Columnas esperadas no encontradas en 'data/dataset.csv'. Se esperan columnas como 'text' y 'label'.")
			print(f"Columnas encontradas: {list(df.columns)}")
			return

		# Renombrar columnas a nombres genéricos
		df_no_header.columns = [f'col{i}' for i in range(len(df_no_header.columns))]

		# Intentar detectar la columna de etiqueta buscando valores comunes
		label_candidates = {'positive', 'negative', 'neutral', 'irrelevant', 'pos', 'neg', 'neu'}
		detected_label = None
		for col in df_no_header.columns:
			vals = df_no_header[col].dropna().astype(str).str.lower().unique()
			if len(set(vals) & label_candidates) >= 1:
				detected_label = col
				break

		# Intentar detectar la columna de texto: la que tenga la mayor longitud media de strings
		detected_text = None
		text_scores = {}
		for col in df_no_header.columns:
			# considerar solo columnas de tipo objeto/string
			try:
				lengths = df_no_header[col].astype(str).map(len)
				text_scores[col] = lengths.mean()
			except Exception:
				text_scores[col] = 0
		if text_scores:
			detected_text = max(text_scores, key=text_scores.get)

		if detected_text is None or detected_label is None:
			print("No pude detectar automáticamente las columnas 'text' y 'label'. Comprueba 'data/dataset.csv'.")
			print(f"Columnas encontradas: {list(df_no_header.columns)}")
			return

		# Usar el dataframe sin encabezado con las columnas detectadas
		text_col = detected_text
		label_col = detected_label
		df = df_no_header

	# Preprocesamiento (esto es un ejemplo, puedes tener un paso de limpieza aquí)
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(df[text_col].astype(str))
	y = df[label_col]

	# Entrenar el modelo
	model = MultinomialNB()
	model.fit(X, y)

	# Asegurar que la carpeta models existe
	os.makedirs('models', exist_ok=True)

	# Guardar el modelo y el vectorizador en la carpeta models
	joblib.dump(model, 'models/model.joblib')
	joblib.dump(vectorizer, 'models/vectorizer.joblib')
	print("¡Modelo y vectorizador guardados en la carpeta models!")


if __name__ == '__main__':
	main()