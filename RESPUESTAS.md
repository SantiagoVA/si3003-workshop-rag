# Respuestas del workshop RAG — SI3003
**Estudiante:** Santiago Vásquez Arboleda (1000357979) 
**Repositorio:** [SantiagoVA/si3003-workshop-rag](https://github.com/SantiagoVA/si3003-workshop-rag)  
**Fecha:** 2026-10-23  

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_
Simulación en Python ejecutada con $M = 16$ y $N = 1.000.000$:
```python
import numpy as np

M = 16
mL = 1 / np.log(M)
N = 1_000_000
rng = np.random.default_rng(0)
l = np.floor(-np.log(1 - rng.random(N)) * mL).astype(int)

for k in range(1, 4):
    print(f"capa >= {k}: simulado={(l >= k).sum():>7,}  teórico={N / M**k:>9,.0f}")
```
Salida obtenida:
```text
capa >= 1: simulado= 62,393  teórico=   62,500
capa >= 2: simulado=  3,912  teórico=    3,906
capa >= 3: simulado=    249  teórico=      244
```

_Mi respuesta / análisis:_
1. **Cálculo a mano:**  
   Dado que la probabilidad de que un nodo alcance la capa $\ge k$ es $P(\text{capa} \ge k) = M^{-k} = 16^{-k}$:
   * **Capa $\ge 1$:** $1.000.000 / 16^1 = 62.500$ nodos.
   * **Capa $\ge 2$:** $1.000.000 / 16^2 = 3.906,25 \approx 3.906$ nodos.
   * **Capa $\ge 3$:** $1.000.000 / 16^3 = 244,14 \approx 244$ nodos.
2. **Comparación:** Los valores simulados ($62.393$, $3.912$ y $249$) difieren en menos del $0,2\%$ respecto a los valores teóricos, confirmando el comportamiento exponencial de la distribución de capas.
3. **Explicación sobre los hubs en capas altas:**  
   Tener muy pocos hubs en las capas superiores permite realizar "saltos largos" a través del espacio vectorial con un número mínimo de cálculos de distancia durante la fase de descenso inicial. Esto reduce drásticamente el espacio de búsqueda y garantiza una complejidad de consulta $O(\log N)$ antes de entrar a la capa 0 densa.


## T2 · ef, k y el tamaño de la cola

_Evidencia:_
Análisis sobre la dinámica de búsqueda en HNSW en capa 0.

_Mi respuesta / análisis:_
1. **Si $k = 10$ y $ef = 5$, ¿qué falla y por qué?**  
   Falla la recuperación porque $ef$ define la capacidad máxima de la cola de prioridad (heap) de candidatos dinámicos durante la búsqueda. Si la cola solo retiene 5 candidatos, el algoritmo no puede devolver 10 vecinos; además, al podar la cola prematuramente, se descartan rutas promisorias hacia vecinos cercanos.
2. **¿Con qué valor conviene empezar y cuándo se sube?**  
   Conviene empezar con $ef = k$ (o un valor prudente entre 16 y 32) como línea base rápida. Se sube cuando la métrica de evaluación (*recall@k*) está por debajo de los requerimientos de la aplicación (por ejemplo, si se necesita recall $\ge 0.95$).
3. **¿Cuál parámetro se puede cambiar en producción sin reconstruir el índice?**  
   Se puede cambiar `ef` (o `efSearch`). Este parámetro únicamente gobierna la profundidad de exploración del grafo en tiempo de consulta (runtime). Por el contrario, `ef_construction` determina las conexiones físicas y aristas del grafo durante el indexado; modificarlo exige reconstruir todo el índice vectorial.


## T3 · Inserción frente a búsqueda

_Evidencia:_
Análisis comparativo de operaciones sobre grafos HNSW.

_Mi respuesta / análisis:_

| Aspecto | Inserción (`add_documents`) | Búsqueda (`retriever.invoke`) | Por qué |
|---|---|---|---|
| **¿Se sortea la capa?** | Sí | No | En inserción se debe definir el nivel máximo hasta donde vivirá el nodo; en búsqueda se empieza siempre en el punto de entrada superior existente. |
| **Valor de ef en capas altas** | 1 | 1 | Ambas usan búsqueda codiciosa (greedy, $ef=1$) para transitar rápidamente entre capas altas hacia la capa objetivo. |
| **Parámetro de la fase principal** | `ef_construction` | `efSearch` / `ef` | `ef_construction` busca vecinos para crear aristas bidireccionales de alta calidad; `efSearch` explora candidatos para encontrar los $k$ más similares. |
| **¿Modifica el índice?** | Sí | No | La inserción añade el vector y actualiza las listas de adyacencia de sus vecinos; la búsqueda es una operación de solo lectura. |

* En el notebook 03, `vectorstore.add_documents(...)` dispara la **Inserción**, mientras que `retriever.invoke(...)` dispara la **Búsqueda**.


## T4 · El invariante de FAISSVectorStore

_Evidencia:_
Trazabilidad de índices posicionales en listas contiguas vs. índices FAISS.

_Mi respuesta / análisis:_

| Posición FAISS | Documento Original | Documento tras borrar B en lista | Comportamiento en búsqueda `[1, 2, 3]` |
|:---:|:---:|:---:|---|
| 0 | A | A | Coincide correctamente. |
| 1 | B | C | **Error silencioso:** FAISS cree retornar B, pero la lista devuelve C. |
| 2 | C | D | **Error silencioso:** FAISS cree retornar C, pero la lista devuelve D. |
| 3 | D | *(No existe)* | **Excepción `IndexError`:** Se intenta acceder a la posición 3 en una lista de tamaño 3. |

* **Relación con bases vectoriales como Qdrant:**  
  FAISS maneja arreglos compactos basados en índices enteros posicionales ($0, 1, \dots, N-1$). Si la capa de aplicación muta los metadatos o documentos sin reindexar FAISS, se rompe el invariante de correspondencia 1:1. Qdrant resuelve esto de forma nativa almacenando los metadatos (*payload*) junto con el vector y asignando identificadores UUID o enteros inmutables a cada punto, garantizando integridad referencial ante eliminaciones y actualizaciones.


## T5 · El costo de la fuerza bruta

_Evidencia:_
Cálculo de complejidad computacional en $1.000.000$ de vectores en $\mathbb{R}^{768}$.

_Mi respuesta / análisis:_
1. **Multiplicaciones por segundo:**  
   $$100 \text{ consultas/s} \times 1.000.000 \text{ vectores} \times 768 \text{ multiplicaciones} = 76.800.000.000 \text{ mult/s}$$
2. **Notación científica:**  
   $$7.68 \times 10^{10} \text{ operaciones/s} \approx 76.8 \text{ GFLOPS}$$  
   Ser exacto (*Flat L2* / fuerza bruta) deja de ser viable porque requiere escanear linealmente toda la memoria RAM en cada consulta ($O(N \cdot d)$). A escala de producción (cientos de consultas concurrentes y millones de vectores), el ancho de banda de memoria y la latencia superan los cientos de milisegundos, haciendo imposible una experiencia en tiempo real.
3. **Compromiso en ANN:**  
   En Approximate Nearest Neighbor (ANN) se sacrifica una pequeña fracción de exactitud ($1\%$ a $5\%$ de recall) a cambio de una reducción exponencial en latencia y costo computacional, pasando de $O(N)$ a $O(\log N)$.


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_
* Captura de la interfaz Gradio con respuesta y fuentes: [`evidencias/A1_gradio.png`](file:///Users/santiago/Downloads/plantilla-entrega/evidencias/A1_gradio.png)
* Notebook ejecutado: [`notebooks/02b_rag_app_nvidia_nim_colab.ipynb`](file:///Users/santiago/Downloads/plantilla-entrega/notebooks/02b_rag_app_nvidia_nim_colab.ipynb)

_Mi respuesta / análisis:_

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas |
|:---:|---|---|---|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise cuesta $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Hasta 3 días por semana, con aprobación del jefe directo. | Política de trabajo remoto |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Los reembolsos mayores a $100 requieren aprobación del supervisor; se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding |
| 5 | ¿Cuál es el procedimiento para solicitar vacaciones pagadas? | No tengo esa información en los documentos proporcionados. | *(Ninguna supera el threshold de 0.3)* |

* **Observación sobre pregunta fuera de contexto:** Cuando la pregunta no tiene relación con los documentos, la similitud coseno cae por debajo del umbral (`score_threshold = 0.30`). El retriever entrega una lista vacía y el LLM responde reconociendo su falta de información, evitando alucinaciones.


## A2 · Tu propio corpus de documentos

_Evidencia:_
* Dataset propio: [`data/company_documents.json`](file:///Users/santiago/Downloads/plantilla-entrega/data/company_documents.json) (8 documentos de NovaTech Solutions con metadatos `department` y `type`).
* Captura de Gradio con el nuevo corpus: [`evidencias/A2_corpus.png`](file:///Users/santiago/Downloads/plantilla-entrega/evidencias/A2_corpus.png)

_Mi respuesta / análisis:_

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas | ¿Correcta? |
|:---:|---|---|---|:---:|
| 1 | ¿Cuál es el tiempo de respuesta garantizado para incidentes de severidad 1 en soporte? | El tiempo de primera respuesta garantizado es menor a 15 minutos [Acuerdos de nivel de servicio (SLA) de soporte técnico]. El tiempo objetivo de mitigación es de 4 horas. | Acuerdos de nivel de servicio (SLA) de soporte técnico | Sí |
| 2 | ¿Qué porcentaje de la prima del seguro médico privado cubre la empresa? | NovaTech Solutions cubre el 80% del valor de la prima del seguro médico privado para el empleado y sus beneficiarios [Beneficios de salud y bienestar integral]. | Beneficios de salud y bienestar integral | Sí |
| 3 | ¿A quién se debe solicitar aprobación para otorgar un descuento del 20%? | Los descuentos comerciales superiores al 15% requieren aprobación formal del gerente comercial [Política de descuentos comerciales]. | Política de descuentos comerciales | Sí |
| 4 | ¿Qué modelo de computador y memoria RAM reciben los ingenieros de software? | A cada ingeniero se le asigna un portátil de alta gama como MacBook Pro de 16 pulgadas o equivalente con 32 GB de RAM [Asignación y ciclo de vida de equipos de cómputo]. | Asignación y ciclo de vida de equipos de cómputo | Sí |
| 5 | ¿Cuántos días de trabajo remoto se permiten y cuál es la franja de conexión obligatoria? | Se permiten hasta 3 días por semana de forma remota y la franja de conexión obligatoria es de 10:00 a 16:00 horas [Política de trabajo remoto y horario flexible]. | Política de trabajo remoto y horario flexible | Sí |
| 6 | ¿Cuál es el procedimiento para solicitar días de vacaciones por maternidad? | No encuentro esa información en los documentos. | *(Ninguna con score $\ge 0.3$)* | Sí |
| 7 | ¿Cuánto es el presupuesto anual para compra de acciones de la empresa? | No encuentro esa información en los documentos. | *(Ninguna con score $\ge 0.3$)* | Sí |

* **Ajuste de `chunk_size`:** Se mantuvo en 400 caracteres con solapamiento de 40. Dado que cada política corporativa es concisa (entre 250 y 380 caracteres), un `chunk_size` de 400 permite que cada fragmento contenga la política completa sin truncar cláusulas condicionales.


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_
Diff del prompt del sistema en `build_pipeline`:
```diff
  prompt = ChatPromptTemplate.from_messages([
      (
          "system",
-         "Eres un asistente empresarial. Responde únicamente con información "
-         "del contexto proporcionado. Si el contexto no contiene la respuesta, "
-         "indica que no tienes esa información. Responde en español y de forma concisa.",
+         "Eres un asistente empresarial. Responde únicamente con información "
+         "del contexto proporcionado. Cita siempre el título del documento entre "
+         "corchetes, por ejemplo [Política de reembolsos]. Si el contexto no alcanza, "
+         "responde exactamente: «No encuentro esa información en los documentos.» "
+         "Responde en español, en máximo 3 oraciones.",
      ),
      ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"),
  ])
```

_Mi respuesta / análisis:_

| Tipo de Pregunta | Pregunta | Respuesta ANTES | Respuesta DESPUÉS |
|---|---|---|---|
| **Fácil (directa)** | ¿Qué modelo de computador y memoria RAM reciben los ingenieros? | Los ingenieros reciben un portátil MacBook Pro de 16 pulgadas o equivalente con 32 GB de RAM y procesador multi-core. | A cada ingeniero de software se le asigna un portátil MacBook Pro de 16 pulgadas o equivalente [Asignación y ciclo de vida de equipos de cómputo]. Este equipo cuenta con 32 GB de RAM y procesador multi-core [Asignación y ciclo de vida de equipos de cómputo]. |
| **Ambigua** | ¿Qué beneficios de salud y dinero adicional ofrece la empresa? | La empresa ofrece seguro médico y dental privado al 80% y un estipendio de $600 anuales para gimnasio y bienestar. | NovaTech Solutions cubre el 80% de la prima del seguro médico y dental [Beneficios de salud y bienestar integral]. Además, asigna un estipendio anual de $600 USD para bienestar físico y salud mental [Beneficios de salud y bienestar integral]. |
| **Fuera de contexto** | ¿Cuál es el presupuesto anual para compra de acciones? | No tengo información sobre presupuesto de compra de acciones en los documentos. | «No encuentro esa información en los documentos.» |

* **Conclusión:** La instrucción con mayor efecto fue la exigencia de la **frase fija exacta** («No encuentro esa información en los documentos.»), ya que eliminó cualquier variación verbosa del modelo cuando el contexto es vacío. La citación entre corchetes `[Título]` funcionó de manera consistente, otorgando trazabilidad inmediata a cada afirmación.


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_
* Captura de pantalla de la interfaz Gradio con scores en vivo: [`evidencias/A4_scores.png`](file:///Users/santiago/Downloads/plantilla-entrega/evidencias/A4_scores.png)
* Implementación en `chat()` con `vectorstore.similarity_search_with_score(question, k=3)`.

_Mi respuesta / análisis:_

| # | Pregunta | Fuente Top 1 y Score | Fuente Top 2 y Score | Fuente Top 3 y Score |
|:---:|---|---|---|---|
| 1 | Incidentes críticos soporte (buena) | **[SLA de soporte técnico]: 0.702** | [SLA soporte (chunk 2)]: 0.474 | [Protocolo de contraseñas]: 0.091 |
| 2 | Portátil y RAM ingenieros (buena) | **[Asignación de equipos]: 0.580** | [Propiedad intelectual]: 0.145 | [Beneficios de salud]: 0.135 |
| 3 | Aprobación descuento 20% (buena) | **[Política de descuentos]: 0.508** | [Política de reembolsos]: 0.126 | [Asignación de equipos]: 0.070 |
| 4 | Solicitud de vacaciones maternidad (fuera) | [Trabajo remoto]: 0.133 | [Reembolsos y viáticos]: 0.130 | [Protocolo de contraseñas]: 0.105 |
| 5 | Presupuesto compra de acciones (fuera) | [Beneficios de salud]: 0.165 | [Asignación de equipos]: 0.149 | [Reembolsos y viáticos]: 0.125 |

* **Umbral propuesto (`score_threshold`):**  
  Se propone fijar `score_threshold = 0.30` (o `0.35`).  
  **Justificación:** Existe una separación clara entre las preguntas legítimas (cuyo chunk relevante obtiene scores entre **0.508 y 0.702**) y las preguntas fuera de contexto o irrelevantes (cuyos mejores scores se ubican entre **0.070 y 0.165**). Un umbral de 0.30 garantiza que ningún fragmento espurio contamine el prompt en preguntas no relacionadas, sin riesgo de descartar documentos pertinentes.


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_
* Notebook ejecutado: [`notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`](file:///Users/santiago/Downloads/plantilla-entrega/notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb)
* 150 chunks indexados en la colección `academic_papers_nvidia` de Qdrant.

_Mi respuesta / análisis:_

| # | Pregunta (resumen) | Respuesta del RAG (resumen) | ¿Correcta? | ¿Paper correcto? |
|:---:|---|---|:---:|:---:|
| **Q1** | BLEU del Transformer big en WMT14 En→De | Obtuvo un score BLEU de **28.4** en WMT 2014 English-to-German. | Sí | Sí (`attention_paper.pdf`) |
| **Q2** | Combinación de memoria paramétrica y no-paramétrica en RAG | Combina memoria paramétrica (seq2seq transformer pre-entrenado) con no-paramétrica (índice denso de vectores de Wikipedia accesible con DPR). | Sí | Sí (`rag_paper.pdf`) |
| **Q3** | Propósito del factor de escala $\sqrt{d_k}$ en atención | Contrarresta el crecimiento en magnitud de los productos punto para $d_k$ grandes, evitando que softmax caiga en gradientes casi nulos. | Sí | Sí (`attention_paper.pdf`) |
| **Q4** | Dos variantes de RAG y diferencias | RAG-Sequence (utiliza el mismo documento para toda la secuencia de salida) y RAG-Token (puede condicionar cada token en un documento distinto). | Sí | Sí (`rag_paper.pdf`) |
| **Q5** | Cabezas de atención y dimensiones base | Emplea $h=8$ cabezas paralelas con dimensión $d_k = d_v = 64$ por cabeza ($d_{\text{model}} / h = 512 / 8$). | Sí | Sí (`attention_paper.pdf`) |
| **Bonus** | Modelo generador de RAG y relación con Transformer | RAG usa BART como generador; las fuentes recuperaron ambos papers demostrando conexión entre retrieval y atención. | Sí | Sí (Ambos papers) |

* **Detalle de la pregunta Bonus:**  
  La consulta recuperó 4 fragmentos de `rag_paper.pdf` (donde se describe el generador BART y el uso de FAISS) y 1 fragmento de `attention_paper.pdf` (arquitectura Transformer).


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_
Barrido experimental sistemático ejecutado mediante las funciones `build_store` y `build_chain`:

_Mi respuesta / análisis:_

| Exp | Parámetro variado | Configuración | # Chunks | Aciertos (de 5) | Observación |
|:---:|:---:|:---:|:---:|:---:|---|
| 1.1 | `chunk_size` | 400 (overlap 50) | 326 | 4 / 5 | Fragmentos muy pequeños rompen ecuaciones matemáticas y tablas explicativas; Q3 pierde parte del contexto. |
| 1.2 | `chunk_size` | 900 (overlap 120) | 150 | 5 / 5 | Configuración óptima: captura párrafos conceptuales completos sin saturar la ventana de contexto. |
| 1.3 | `chunk_size` | 1500 (overlap 200) | 96 | 5 / 5 | Acierta todas, pero diluye la densidad semántica del vector y triplica el costo de tokens en el prompt. |
| 2.1 | `k` | 3 (en store 900) | 150 | 5 / 5 | Recuperación muy limpia y precisa; menor consumo de tokens y latencia más baja. |
| 2.2 | `k` | 5 (en store 900) | 150 | 5 / 5 | Línea base del workshop; balance sólido entre cobertura y especificidad. |
| 2.3 | `k` | 8 (en store 900) | 150 | 5 / 5 | Introduce fragmentos secundarios con scores marginales sin aportar valor factual nuevo. |
| 3.1 | `threshold` | 0.15 (en store 900, k=5) | 150 | 5 / 5 | Pasa todo el top-5; riesgoso ante preguntas ambiguas por admitir ruido. |
| 3.2 | `threshold` | 0.25 (en store 900, k=5) | 150 | 5 / 5 | Umbral equilibrado; filtra ruido manteniendo fragmentos relevantes de ambos papers. |
| 3.3 | `threshold` | 0.40 (en store 900, k=5) | 150 | 4 / 5 | **Demasiado restrictivo:** descarta fragmentos cruciales de Q3 cuyo score es $\approx 0.38$, degradando la respuesta. |

* **Conclusión (5 a 8 líneas):**  
  El parámetro con mayor impacto sobre la calidad final fue `score_threshold`, seguido de cerca por `chunk_size`. Mientras que variar $k$ entre 3 y 8 mantuvo los aciertos casi invariantes (solo afectando el volumen de tokens), elevar el umbral a $0.40$ provocó la pérdida directa de evidencia relevante en preguntas técnicas donde la similitud coseno ronda $0.35$ a $0.38$. Por otro lado, un `chunk_size` de 400 fragmentó excesivamente definiciones matemáticas densas, mientras que 1500 agregó información superflua. La mejor configuración encontrada para papers académicos es `chunk_size = 900`, `overlap = 120`, $k = 4$ o $5$ y `score_threshold = 0.25`.


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_
Evaluación de honestidad epistémica ante preguntas formuladas sobre literatura post-2020:

_Mi respuesta / análisis:_

| # | Pregunta trampa | Respuesta del RAG | Clasificación |
|:---:|---|---|---|
| 1 | *What was the perplexity achieved by GPT-4 on the Attention Is All You Need dataset?* | *The excerpts do not contain enough evidence. (Attention Is All You Need)* | **Admite que no sabe** |
| 2 | *How does the Llama-3 architecture optimize the RAG-Token generator component?* | *The provided excerpts do not contain information about how the Llama-3 architecture optimizes the RAG-Token generator component. (rag_paper.pdf)* | **Admite que no sabe** |
| 3 | *What learning rate was used to train the Vision Transformer (ViT) in the Attention paper?* | *The excerpts do not contain enough evidence about the learning rate used to train the Vision Transformer (ViT). (Attention Is All You Need)* | **Admite que no sabe** |

* **Análisis:** En las tres preguntas, a pesar de que el retriever devolvió chunks porque ciertas palabras clave coincidían (*"perplexity"*, *"RAG-Token"*, *"learning rate"*), el prompt reforzado en LCEL forzó al modelo a evaluar la suficiencia de la evidencia, admitiendo la ausencia de datos en lugar de inventar cifras plausibles.


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_
Traducción y ejecución de Q1, Q3 y Q4 consultadas directamente en español:

_Mi respuesta / análisis:_

| Pregunta | Chunks recuperados (EN vs ES) | Calidad de la respuesta en español |
|---|---|---|
| **Q1_es** (BLEU de Transformer big) | Recuperó exactamente los mismos 5 chunks de `attention_paper.pdf`. | **Excelente:** Respondió $28.4$ BLEU citando el paper en perfecto español. |
| **Q3_es** (Propósito de factor $\sqrt{d_k}$) | Recuperó los mismos 4 chunks principales de `attention_paper.pdf`. | **Excelente:** Explicó el crecimiento de magnitud y el aplanamiento de gradientes en softmax. |
| **Q4_es** (Variantes RAG-Sequence y Token) | Recuperó los mismos 5 chunks de `rag_paper.pdf`. | **Excelente:** Distinguió con precisión entre condicionamiento por secuencia completa vs. por token individual. |

* **Hipótesis sobre búsqueda entre idiomas:**  
  El modelo `nemotron-3-embed-1b` fue pre-entrenado sobre corpus masivos multilingües, lo que proyecta conceptos semánticos equivalentes en distintos idiomas a regiones hiperespaciales prácticamente contiguas. Para comprobar esta hipótesis formalmente, se puede calcular la similitud coseno directa entre $\vec{v}_{\text{query\_en}}$ y $\vec{v}_{\text{query\_es}}$: si la similitud supera $0.85$, se demuestra empíricamente la alineación cross-lingual del espacio vectorial.


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_
Nuevas preguntas con ground truth extraído directamente de los PDFs e incorporadas al notebook 03:

_Mi respuesta / análisis:_

| # | Pregunta | Ground Truth y Sección | Respuesta del RAG | ¿Correcta? |
|:---:|---|---|---|:---:|
| **NQ1** | *What optimizer, learning rate schedule, and warmup steps were used to train the Transformer models?* | Optimizador Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$), warmup de 4.000 pasos y decaimiento inversamente proporcional a la raíz cuadrada del paso. *(Attention Section 5.3)* | Identificó los hiperparámetros de Adam ($\beta_1=0.9, \beta_2=0.98$), la fórmula de decaimiento y los 4000 pasos de warmup. | Sí |
| **NQ2** | *What Wikipedia dump was used for non-parametric retrieval in RAG, and into how many passages was it split?* | Volcado de Wikipedia de diciembre de 2018, dividido en 21 millones de fragmentos disjuntos de 100 palabras. *(RAG Section 2.1)* | Indicó claramente el dump de diciembre de 2018 y los 21 millones de fragmentos de 100 palabras. | Sí |
| **NQ3 (Cruzada)** | *How does the masking in the Transformer decoder self-attention ensure autoregressive generation, and why is this property essential for the BART generator in RAG?* | El enmascaramiento en la auto-atención del decodificador impide atender a tokens futuros, garantizando causalidad paso a paso ($< i$). Esto es indispensable para BART en RAG porque genera la respuesta token a token condicionada en el contexto recuperado. *(Attention Sec 3.2.3 y RAG Sec 2.1)* | Explicó el enmascaramiento causal en el decodificador de Attention y su relación con la generación autorregresiva de BART en RAG. | Sí |


---

# Reflexión

## R1 · Reflexión final con tus datos

_Mi respuesta / análisis:_

1. **¿Qué ocurrió cuando el contexto no contenía información suficiente?**  
   Cuando la información no estuvo presente en los documentos (como en las preguntas trampa de B3 sobre GPT-4 o Llama-3, y en las consultas fuera de dominio del corpus empresarial en A2), el sistema demostró un comportamiento determinista y seguro. Si ningún documento superaba el umbral de similitud ($0.25$ o $0.30$), el retriever entregó un contexto vacío y el LLM respondió con la negativa estandarizada exigida por el prompt. En los casos donde se recuperaron fragmentos con similitudes marginales pero no pertinentes, la restricción del system prompt evitó la alucinación, logrando que el modelo reportara explícitamente que la evidencia disponible era insuficiente.

2. **¿Qué fragmentos recuperó para la pregunta bonus? ¿Aparecieron ambos papers?**  
   Para la pregunta bonus se recuperaron 5 fragmentos en total: 4 pertenecientes a `rag_paper.pdf` y 1 a `attention_paper.pdf`. Esta distribución refleja exactamente la naturaleza de la consulta: la descripción del generador BART y el uso de FAISS provienen del paper de RAG, mientras que la formulación fundamental de la arquitectura Transformer reside en el paper de Vaswani et al. Gracias a la inclusión del título y archivo en los metadatos de cada chunk (`[title | source]`), el LLM pudo correlacionar la información de ambos documentos sin confundir las atribuciones bibliográficas.

3. **¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?**  
   En nuestras pruebas de la Tarea B2, la fragmentación en `chunk_size = 400` degradó el acierto a 4/5 porque cortó definiciones que requerían oraciones contiguas, mientras que `chunk_size = 1500` incrementó el costo de inferencia sin aportar mejoras en exactitud. El parámetro $k$ demostró que valores entre 3 y 5 son ideales; aumentar $k$ a 8 solo introdujo pasajes secundarios que añadieron redundancia. Finalmente, `score_threshold` demostró ser el filtro más sensible: un umbral de $0.25$ fue óptimo, mientras que elevarlo a $0.40$ causó falsos negativos al filtrar fragmentos matemáticos relevantes cuyo score promedio fue de $0.381$.

4. **¿Por qué conviene separar el modelo de embeddings del modelo generativo?**  
   Conviene desacoplarlos por razones de especialización arquitectónica, costos y escalabilidad computacional. El modelo de embeddings (`nemotron-3-embed-1b`) está optimizado exclusivamente para mapear semántica densa en espacios vectoriales compactos de forma bidireccional y de bajísima latencia, permitiendo indexar millones de documentos de manera eficiente. En contraste, el modelo generativo (LLM de 30B a 120B parámetros) es autorregresivo, computacionalmente costoso y está diseñado para razonamiento, síntesis y seguimiento de instrucciones. Desacoplarlos permite escalar el almacenamiento vectorial y la búsqueda independientemente del cómputo de generación.

5. **¿Qué cambiarías para llevar este prototipo a producción?**  
   Para migrar este prototipo a un entorno productivo empresarial se requerirían cinco pilares fundamentales:
   * **Persistencia y escalabilidad:** Reemplazar `QdrantClient(":memory:")` por un clúster distribuido de Qdrant en Kubernetes con almacenamiento persistente en disco SSD, réplicas de lectura y particionamiento (*sharding*).
   * **Control de acceso basado en roles (RBAC):** Integrar filtros de metadatos a nivel de vectorstore para que un usuario solo pueda recuperar documentos según sus privilegios organizacionales (ej. departamento, nivel de seguridad).
   * **Reranking y arquitectura híbrida:** Incorporar búsqueda híbrida (Dense Retrieval + BM25 léxico) complementada con un modelo Cross-Encoder de Re-ranking (como Cohere Rerank o NVIDIA NeMo Retriever) antes de enviar los fragmentos al LLM.
   * **Observabilidad y evaluación continua:** Implementar trazabilidad de llamadas mediante LangSmith o OpenTelemetry para auditar latencias, consumo de tokens, calidad de recuperación (*Hit@K, MRR*) y detección de alucinaciones en tiempo real.
   * **Estrategias de caching y costos:** Incorporar una capa de Semantic Cache (Redis / GPTCache) para responder consultas recurrentes sin consultar la API de embeddings ni el LLM, reduciendo costos operativos en más de un 40%.
