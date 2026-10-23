# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Santiago
- **Código estudiantil:** [Tu código aquí]
- **Usuario de GitHub:** [SantiagoVA](https://github.com/SantiagoVA)
- **Repositorio:** [si3003-workshop-rag](https://github.com/SantiagoVA/si3003-workshop-rag)
- **Fecha:** 23 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#t1--cuántos-nodos-viven-en-cada-capa) |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#t2--ef-k-y-el-tamaño-de-la-cola) |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#t3--inserción-frente-a-búsqueda) |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#t4--el-invariante-de-faissvectorstore) |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#t5--el-costo-de-la-fuerza-bruta) |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | [evidencias/A1_gradio.png](evidencias/A1_gradio.png) · [RESPUESTAS.md](RESPUESTAS.md#a1--ejecuta-la-app-rag-de-punta-a-punta) |
| A2 | Tu propio corpus de documentos | 6 | ✅ | [data/company_documents.json](data/company_documents.json) · [evidencias/A2_corpus.png](evidencias/A2_corpus.png) |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#a3--ingeniería-del-prompt-antes-y-después) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | [evidencias/A4_scores.png](evidencias/A4_scores.png) · [RESPUESTAS.md](RESPUESTAS.md#a4--muestra-el-score-de-similitud-en-las-fuentes) |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#b1--ejecuta-el-rag-de-papers-y-evalúa-con-el-ground-truth) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#b2--experimentos-con-chunk_size-k-y-score_threshold) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#b3--preguntas-trampa-el-rag-admite-que-no-sabe) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#b4--preguntas-en-español-sobre-papers-en-inglés) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#b5--escribe-tu-propio-quiz-con-ground-truth) |
| R1 | Reflexión final con tus datos | 10 | ✅ | [RESPUESTAS.md](RESPUESTAS.md#r1--reflexión-final-con-tus-datos) |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ⬜ | |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ⬜ | |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ⬜ | |
| C5 | MMR frente a similitud pura (reto) | 8 | ⬜ | |
| C6 | Persistencia del índice en disco (reto) | 8 | ⬜ | |

**Autoevaluación:** 70 / 100 puntos (100% del Núcleo completado con evidencia verificable).

## Cómo ejecutar mis notebooks

### Opción 1: En Google Colab (Prioridad nativa)
1. Sube o abre los notebooks de `notebooks/` en Google Colab:
   - `notebooks/02b_rag_app_nvidia_nim_colab.ipynb`
   - `notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`
2. Configura tu secreto `NVIDIA_API_KEY` en el panel de **Secrets (🔑)** de Colab y habilita el toggle de acceso para el notebook.
3. Ejecuta las celdas en orden.

### Opción 2: Ejecución local (con archivo `.env`)
Los notebooks cuentan con detección automática en cascada (Google Colab Secrets → Archivo `.env` local → `getpass`):
1. Asegúrate de tener tu clave configurada en un archivo `.env` en la raíz:
   ```env
   NVIDIA_API_KEY="nvapi-..."
   ```
2. Ejecuta los notebooks directamente desde tu entorno o IDE preferido (VS Code, JupyterLab, Cursor, etc.).

> **Seguridad:** El archivo `.env` está excluido en `.gitignore` para no exponer credenciales en el repositorio.

## Decisiones y aprendizajes

Durante el workshop se evidenció que la calidad de un sistema RAG no depende únicamente del tamaño del LLM, sino críticamente del diseño de la recuperación vectorial y la ingeniería del prompt. Descubrimos que `score_threshold` es el parámetro más sensible: un umbral muy permisivo admite pasajes irrelevantes que propician alucinaciones en preguntas trampa, mientras que un umbral estricto ($\ge 0.40$) descarta fragmentos técnicos legítimos. Asimismo, comprobamos la efectividad del espacio latente multilingüe de `nemotron-3-embed-1b`, el cual permitió consultar en español documentos técnicos en inglés con idéntica precisión de recuperación. Finalmente, separar el modelo de embeddings de la generación y aplicar citas estructuradas `[Título]` garantiza trazabilidad y honestidad epistémica sin degradar la experiencia de usuario.
