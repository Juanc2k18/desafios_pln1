# Resumen de desafíos de PLN 1  
**Alumno:** Juan Miguel Chunga  

---

## Desafío 1: Clasificación de Textos con 20 Newsgroups

### Vectorización y Similaridad  
- Se aplicó **TfidfVectorizer** para transformar documentos en vectores numéricos.  
- La **similaridad coseno** permitió identificar documentos relacionados, aunque con valores bajos; útil como señal, pero insuficiente por sí sola.

### Clasificación Zero-Shot  
- Se asignó la clase del documento más similar según coseno.  
- El método obtuvo **F1 macro = 0.5050**, inferior a Naïve Bayes, evidenciando limitaciones al basarse solo en similitud.

### Modelos Naïve Bayes  
- Se entrenaron **MultinomialNB** y **ComplementNB**, realizando búsqueda de hiperparámetros.  
- Mejores resultados:  
  - **MultinomialNB:** F1 macro = 0.6753  
    - `min_df=2`, `max_df=0.7`, `max_features=20000`, `alpha=0.1`  
  - **ComplementNB:** F1 macro = 0.6980  
    - `min_df=2`, `max_df=0.9`, `max_features=None`, `alpha=0.5`  
- Ambos superaron ampliamente al método zero-shot.

### Similaridad entre Palabras  
- Se transpusó la matriz documento-término para obtener una matriz término-documento.  
- Las palabras seleccionadas mostraron asociaciones semánticas coherentes pese a similitudes bajas.

---

## Desafío 2: Análisis de Similaridad entre Palabras en Letras de un Artista  
- Se limpiaron los textos removiendo *stopwords* y palabras muy cortas.  
- La proyección 2D mostró grupos por idioma y por canción.  
- Los términos en español evidencian pertenencia a una canción que no deberia ser parte del dataset, aunque requiere análisis adicional.

---

## Desafío 3: Generación de Texto con Modelos de Lenguaje

### Preprocesamiento  
- Corpus: *El Caballero Carmelo y otros cuentos*. Anteriormente, se intentó con *El Conde de Montecristo*.
- Normalización a minúsculas, tokenización a nivel de carácter y construcción de índices.  
- Generación de secuencias con ventanas de contexto y estructuración del dataset.

### Arquitecturas Evaluadas  
- Modelos: **SimpleRNN**, **LSTM** (GRU mencionado).  
- Uso de `TimeDistributed` + one-hot + capa recurrente + `Dense` con `softmax`.  
- Pérdida: `sparse_categorical_crossentropy`; optimizador: `rmsprop`.

### Entrenamiento  
- Implementación de **callback de perplejidad** con *early stopping*.  
- Ajustes necesarios por limitaciones de memoria GPU.  
- Se visualizaron métricas de pérdida y perplejidad.

### Decodificación  
- **Greedy search** y **beam search** (determinístico y estocástico).  
- Integración con **Gradio** para predicción interactiva.

### Conclusiones
- El trabajo no pudo ser concluido por limitaciones tecnicas.

---

## Desafío 4: Traducción Automática  
- Se entrenó un modelo de traducción en tres iteraciones.  
- Las métricas de accuracy no reflejaron adecuadamente la calidad.  
- Más unidades mejoraron el ajuste (iteración 2), menos unidades degradaron la calidad (iteración 3).  
- Limitaciones claras por tamaño reducido del dataset y mecanismos de decodificación simples.

---

## Limitaciones Técnicas  
- Múltiples ajustes por incompatibilidades y depreciaciones en librerías.
- Codigo base de Pytorch con errores, se reformuló, pero fue necesario migrar a la version de TensorFlow.
- Reducción necesaria del dataset, disminuyendo la capacidad de aprendizaje. 
- Alto consumo de memoria:  
  - ~8 GB desde la época 1  
  - ~8.7 GB hacia la época 15  
- Tiempos por dispositivo:  
  - CPU: ~400 ms/step  
  - GPU L4: ~16 ms/step  

---

## Observaciones sobre Consistencia  
- Los problemas de comportamiento observados en PyTorch reaparecen en esta implementación.  
- Las causas principales parecen ser:  
  - Pocos datos,  
  - Arquitectura simple,  
  - Decodificación limitada.

---

## Conclusiones Generales  
Para mejorar el desempeño en traducción:  
- Utilizar el dataset completo.  
- Implementar regularización para evitar sobreajuste.  
- Incorporar **transfer learning** con modelos preentrenados.

---
