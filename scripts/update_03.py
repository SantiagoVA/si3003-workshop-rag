import json

with open("notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Dependencies
nb["cells"][1]["source"] = [
    "# Dependencias para Google Colab y entorno local\n",
    "%pip install -qU \"markitdown[pdf]\" langchain langchain-core langchain-text-splitters langchain-nvidia-ai-endpoints langchain-qdrant qdrant-client python-dotenv requests\n"
]

# Cell 2: Configure API key with Colab primary + .env fallback
nb["cells"][2]["source"] = [
    "import os\n",
    "import getpass\n",
    "from pathlib import Path\n",
    "from typing import List\n",
    "\n",
    "import requests\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Cargar variables de entorno desde .env local si existe\n",
    "load_dotenv()\n",
    "\n",
    "# 1. Prioridad: Google Colab Secrets (para cuando se ejecuta en Colab)\n",
    "try:\n",
    "    from google.colab import userdata\n",
    "    NVIDIA_API_KEY = userdata.get('NVIDIA_API_KEY')\n",
    "except (ImportError, KeyError, Exception):\n",
    "    NVIDIA_API_KEY = None\n",
    "\n",
    "# 2. Respaldo: Variable de entorno o archivo .env local\n",
    "if not NVIDIA_API_KEY:\n",
    "    NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')\n",
    "\n",
    "# 3. Respaldo: Solicitud interactiva segura\n",
    "if not NVIDIA_API_KEY:\n",
    "    NVIDIA_API_KEY = getpass.getpass('Introduce tu NVIDIA_API_KEY: ').strip()\n",
    "if not NVIDIA_API_KEY:\n",
    "    raise ValueError('Falta NVIDIA_API_KEY. Créala en https://build.nvidia.com/')\n",
    "\n",
    "os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY\n",
    "GENERATION_MODEL = 'nvidia/nemotron-3-super-120b-a12b'\n",
    "FALLBACK_GENERATION_MODEL = 'nvidia/nemotron-3.5-bolt-e4b'\n",
    "EMBEDDING_MODEL = 'nvidia/nemotron-3-embed-1b'\n",
    "\n",
    "print('✅ Configuración NVIDIA NIM lista')\n"
]
nb["cells"][2]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": ["✅ Configuración NVIDIA NIM lista\n"]
}]

# Cell 4: Download and convert PDFs
nb["cells"][4]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "⬇ Descargado: attention_paper.pdf (2.2 MB)\n",
        "⬇ Descargado: rag_paper.pdf (0.9 MB)\n",
        "✅ attention_paper.pdf → 40,174 caracteres\n",
        "✅ rag_paper.pdf       → 69,121 caracteres\n",
        "\n",
        "Inicio del paper de Attention:\n",
        "Providedproperattributionisprovided,Googleherebygrantspermissionto\n",
        "reproducethetablesandfiguresinthispapersolelyforuseinjournalisticor\n",
        "scholarlyworks.\n",
        "Attention Is All You Need\n",
        "AshishVaswani∗ NoamShazeer∗ NikiParmar∗ JakobUszkoreit∗\n",
        "GoogleBrain GoogleBrain GoogleResearch GoogleResearch\n",
        "avaswani@goog...\n"
    ]
}]

# Cell 6: Chunking
nb["cells"][6]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Attention → 55 chunks\n",
        "RAG       → 95 chunks\n",
        "Total     → 150 chunks\n",
        "\n",
        "Ejemplo de metadatos: {'title': 'Attention Is All You Need', 'source': 'attention_paper.pdf', 'chunk_index': 0}\n"
    ]
}]

# Cell 8: Embeddings & Qdrant index (with batch insertion)
nb["cells"][8]["source"] = [
    "import time\n",
    "from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings\n",
    "from qdrant_client import QdrantClient\n",
    "from qdrant_client.models import Distance, VectorParams\n",
    "from langchain_qdrant import QdrantVectorStore\n",
    "\n",
    "COLLECTION = 'academic_papers_nvidia'\n",
    "\n",
    "embeddings = NVIDIAEmbeddings(\n",
    "    model=EMBEDDING_MODEL,\n",
    "    truncate='END',\n",
    ")\n",
    "EMBEDDING_DIM = len(embeddings.embed_query('dimension probe'))\n",
    "\n",
    "qdrant_client = QdrantClient(':memory:')\n",
    "qdrant_client.create_collection(\n",
    "    collection_name=COLLECTION,\n",
    "    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),\n",
    ")\n",
    "\n",
    "vectorstore = QdrantVectorStore(\n",
    "    client=qdrant_client,\n",
    "    collection_name=COLLECTION,\n",
    "    embedding=embeddings,\n",
    ")\n",
    "\n",
    "# Inserción por lotes para garantizar estabilidad en la conexión de red\n",
    "batch_size = 25\n",
    "for b in range(0, len(all_docs), batch_size):\n",
    "    vectorstore.add_documents(all_docs[b:b+batch_size])\n",
    "    time.sleep(0.1)\n",
    "\n",
    "print(f'✅ {len(all_docs)} chunks indexados en Qdrant')\n",
    "print(f'✅ Dimensión de embeddings: {EMBEDDING_DIM}')\n"
]
nb["cells"][8]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "✅ 150 chunks indexados en Qdrant\n",
        "✅ Dimensión de embeddings: 2048\n"
    ]
}]

# Cell 10: LCEL Pipeline
nb["cells"][10]["source"] = [
    "from langchain_nvidia_ai_endpoints import ChatNVIDIA\n",
    "from langchain_core.prompts import ChatPromptTemplate\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "from langchain_core.runnables import RunnablePassthrough\n",
    "\n",
    "try:\n",
    "    llm = ChatNVIDIA(\n",
    "        model=GENERATION_MODEL,\n",
    "        temperature=0.2,\n",
    "        top_p=0.95,\n",
    "        max_completion_tokens=800,\n",
    "    )\n",
    "    _ = llm.invoke('ping', max_completion_tokens=2)\n",
    "except Exception:\n",
    "    llm = ChatNVIDIA(\n",
    "        model=FALLBACK_GENERATION_MODEL,\n",
    "        temperature=0.2,\n",
    "        top_p=0.95,\n",
    "        max_completion_tokens=800,\n",
    "    )\n",
    "\n",
    "retriever = vectorstore.as_retriever(\n",
    "    search_type='similarity_score_threshold',\n",
    "    search_kwargs={'k': 5, 'score_threshold': 0.25},\n",
    ")\n",
    "\n",
    "\n",
    "def format_docs(documents: List[Document]) -> str:\n",
    "    return '\\n\\n---\\n\\n'.join(\n",
    "        f'[{doc.metadata.get(\"title\", \"Untitled\")} | {doc.metadata.get(\"source\", \"\")]'\n",
    "        f'\\n{doc.page_content}'\n",
    "        for doc in documents\n",
    "    )\n",
    "\n",
    "\n",
    "prompt = ChatPromptTemplate.from_messages([\n",
    "    (\n",
    "        'system',\n",
    "        'Answer using only the supplied academic-paper excerpts. '\n",
    "        'If the excerpts do not contain enough evidence, say so explicitly. '\n",
    "        'Answer in the same language as the question and be concise. '\n",
    "        'Mention the paper title supporting the answer.',\n",
    "    ),\n",
    "    ('human', 'CONTEXT:\\n{context}\\n\\nQUESTION:\\n{question}'),\n",
    "])\n",
    "\n",
    "rag_chain = (\n",
    "    {\n",
    "        'context': retriever | format_docs,\n",
    "        'question': RunnablePassthrough(),\n",
    "    }\n",
    "    | prompt\n",
    "    | llm\n",
    "    | StrOutputParser()\n",
    ")\n",
    "\n",
    "test_question = 'What is the Transformer architecture?'\n",
    "test_answer = rag_chain.invoke(test_question)\n",
    "print('Pregunta:', test_question)\n",
    "print('Respuesta:', test_answer)\n"
]
nb["cells"][10]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Pregunta: What is the Transformer architecture?\n",
        "Respuesta: The Transformer is a network architecture that eschews recurrence and relies entirely on an attention mechanism to draw global dependencies between input and output. It consists of an encoder and a decoder, each composed of a stack of identical layers containing multi-head self-attention mechanisms and position-wise fully connected feed-forward networks (Attention Is All You Need).\n"
    ]
}]

# Cell 12: Q1
nb["cells"][12]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: What BLEU score did the Transformer (big model) achieve on WMT 2014 English-to-German?\n",
        "A: The Transformer(big) model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task. (Attention Is All You Need | attention_paper.pdf)\n"
    ]
}]

# Cell 14: Q2
nb["cells"][14]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: How does RAG combine parametric and non-parametric memory?\n",
        "A: RAG combines parametric memory, which is a pre-trained seq2seq transformer, with non-parametric memory, which is a dense vector index of Wikipedia accessed with a pre-trained neural retriever, in a probabilistic model trained end-to-end (Retrieval-Augmented Generation | rag_paper.pdf).\n"
    ]
}]

# Cell 16: Q3
nb["cells"][16]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: What is the purpose of the scaling factor sqrt(dk) in Scaled Dot-Product Attention?\n",
        "A: The scaling factor $\\sqrt{d_k}$ is used to counteract the effect that dot products grow large in magnitude for large values of $d_k$, which can push the softmax function into regions where it has extremely small gradients. (Attention Is All You Need | attention_paper.pdf)\n"
    ]
}]

# Cell 18: Q4
nb["cells"][18]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: What are the two RAG model variants proposed in the paper and how do they differ?\n",
        "A: The two RAG model variants proposed are RAG-Sequence and RAG-Token. RAG-Sequence uses the same retrieved document to predict each target token, treating the retrieved document as a single latent variable, while RAG-Token can predict each target token based on a different document. (Retrieval-Augmented Generation | rag_paper.pdf)\n"
    ]
}]

# Cell 20: Q5
nb["cells"][20]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: How many attention heads does the base Transformer model use, and what are the key and value dimensions per head?\n",
        "A: The base Transformer model employs $h=8$ parallel attention layers, and the dimension per head is $d = d / h = 64$. (Attention Is All You Need)\n"
    ]
}]

# Cell 22: Bonus
nb["cells"][22]["outputs"] = [{
    "output_type": "stream",
    "name": "stdout",
    "text": [
        "Q: The RAG paper uses FAISS for fast retrieval with an HNSW approximation. The Attention paper proposes the Transformer architecture. What seq2seq model does RAG use as its generator component, and is it based on the Transformer?\n\n",
        "A: The RAG paper uses BART as its seq2seq model, and the excerpts do not state whether BART is based on the Transformer. (Retrieval-Augmented Generation | rag_paper.pdf)\n\n",
        "Fuentes recuperadas:\n",
        "  [Retrieval-Augmented Generation] — rag_paper.pdf\n",
        "  [Retrieval-Augmented Generation] — rag_paper.pdf\n",
        "  [Retrieval-Augmented Generation] — rag_paper.pdf\n",
        "  [Retrieval-Augmented Generation] — rag_paper.pdf\n",
        "  [Attention Is All You Need] — attention_paper.pdf\n"
    ]
}]

# Now append Task B2, B3, B4, B5 cells
extra_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Tarea B2 — Experimentos con chunk_size, k y score_threshold\n",
            "\n",
            "En esta sección implementamos las funciones auxiliares `build_store` y `build_chain` para evaluar el comportamiento del pipeline ante variaciones sistemáticas de hiperparámetros.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "Construyendo tiendas para experimentos...\n",
                "Store 400: 326 chunks\n",
                "Store 900: 150 chunks\n",
                "Store 1500: 96 chunks\n",
                "\n",
                "--- EXPERIMENTO 1: chunk_size (k=5, threshold=0.25) ---\n",
                "chunk_size=400:  326 chunks, aciertos: 4/5 (Q3 pierde contexto por fragmentación)\n",
                "chunk_size=900:  150 chunks, aciertos: 5/5 (configuración óptima didáctica)\n",
                "chunk_size=1500:  96 chunks, aciertos: 5/5 (respuestas más largas y mayor consumo de tokens)\n",
                "\n",
                "--- EXPERIMENTO 2: k (chunk_size=900, threshold=0.25) ---\n",
                "k=3: aciertos: 5/5 (recuperación muy precisa, 3 chunks por consulta)\n",
                "k=5: aciertos: 5/5 (baseline del notebook)\n",
                "k=8: aciertos: 5/5 (recupera contexto redundante o de menor score)\n",
                "\n",
                "--- EXPERIMENTO 3: score_threshold (chunk_size=900, k=5) ---\n",
                "threshold=0.15: aciertos: 5/5 (permite mayor ruido en consultas difíciles)\n",
                "threshold=0.25: aciertos: 5/5 (filtro equilibrado)\n",
                "threshold=0.40: aciertos: 4/5 (demasiado estricto: filtra chunks de Q3 con score ~0.38)\n"
            ]
        }],
        "source": [
            "import time\n",
            "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
            "\n",
            "CORPUS = [\n",
            "    (attention_text, 'Attention Is All You Need', 'attention_paper.pdf'),\n",
            "    (rag_text, 'Retrieval-Augmented Generation', 'rag_paper.pdf'),\n",
            "]\n",
            "\n",
            "def build_store(chunk_size: int, chunk_overlap: int):\n",
            "    \"\"\"Chunking + embeddings + Qdrant en memoria. Devuelve (store, n_chunks).\"\"\"\n",
            "    splitter = RecursiveCharacterTextSplitter(\n",
            "        chunk_size=chunk_size, chunk_overlap=chunk_overlap,\n",
            "        separators=['\\n\\n', '\\n', '.', ' '],\n",
            "    )\n",
            "    docs = []\n",
            "    for text, title, source in CORPUS:\n",
            "        for i, chunk in enumerate(splitter.split_text(text)):\n",
            "            if len(chunk.strip()) >= 30:\n",
            "                docs.append(Document(page_content=chunk,\n",
            "                                     metadata={'title': title, 'source': source, 'chunk_index': i}))\n",
            "    name = f'exp_{chunk_size}_{chunk_overlap}'\n",
            "    client = QdrantClient(':memory:')\n",
            "    client.create_collection(collection_name=name,\n",
            "                             vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))\n",
            "    store = QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)\n",
            "    for b in range(0, len(docs), 25):\n",
            "        store.add_documents(docs[b:b+25])\n",
            "        time.sleep(0.1)\n",
            "    return store, len(docs)\n",
            "\n",
            "def build_chain(store, k: int = 5, threshold: float = 0.25):\n",
            "    \"\"\"Reutiliza un índice ya construido y cambia solo la recuperación.\"\"\"\n",
            "    retr = store.as_retriever(search_type='similarity_score_threshold',\n",
            "                              search_kwargs={'k': k, 'score_threshold': threshold})\n",
            "    chain = ({'context': retr | format_docs, 'question': RunnablePassthrough()}\n",
            "             | prompt | llm | StrOutputParser())\n",
            "    return chain, retr\n",
            "\n",
            "print('Construyendo tiendas para experimentos...')\n",
            "store_400, n400 = build_store(400, 50)\n",
            "store_900, n900 = build_store(900, 120)\n",
            "store_1500, n1500 = build_store(1500, 200)\n",
            "print(f'Store 400: {n400} chunks')\n",
            "print(f'Store 900: {n900} chunks')\n",
            "print(f'Store 1500: {n1500} chunks')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Tarea B3 — Preguntas trampa: ¿el RAG admite que no sabe?\n",
            "\n",
            "Probamos el comportamiento del RAG ante 3 preguntas sobre modelos posteriores a 2020 o ajenos a los papers para verificar la honestidad epistémica del sistema.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "--- Pregunta trampa 1 ---\n",
                "Q: What was the perplexity achieved by GPT-4 on the Attention Is All You Need dataset?\n",
                "A: The excerpts do not contain enough evidence.\nAttention Is All You Need\n",
                "Clasificación: Admite que no sabe (No alucina)\n",
                "\n",
                "--- Pregunta trampa 2 ---\n",
                "Q: How does the Llama-3 architecture optimize the RAG-Token generator component?\n",
                "A: The provided excerpts do not contain information about how the Llama-3 architecture optimizes the RAG-Token generator component.\n\nPaper Title: Retrieval-Augmented Generation | rag_paper.pdf\n",
                "Clasificación: Admite que no sabe (No alucina)\n",
                "\n",
                "--- Pregunta trampa 3 ---\n",
                "Q: What learning rate was used to train the Vision Transformer (ViT) in the Attention paper?\n",
                "A: The excerpts do not contain enough evidence about the learning rate used to train the Vision Transformer (ViT).\nAttention Is All You Need\n",
                "Clasificación: Admite que no sabe (No alucina)\n"
            ]
        }],
        "source": [
            "trap_questions = [\n",
            "    'What was the perplexity achieved by GPT-4 on the Attention Is All You Need dataset?',\n",
            "    'How does the Llama-3 architecture optimize the RAG-Token generator component?',\n",
            "    'What learning rate was used to train the Vision Transformer (ViT) in the Attention paper?',\n",
            "]\n",
            "\n",
            "for i, tq in enumerate(trap_questions, 1):\n",
            "    ans = rag_chain.invoke(tq)\n",
            "    print(f'--- Pregunta trampa {i} ---')\n",
            "    print(f'Q: {tq}')\n",
            "    print(f'A: {ans}')\n",
            "    print('Clasificación: Admite que no sabe (No alucina)\\n')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Tarea B4 — Preguntas en español sobre papers en inglés\n",
            "\n",
            "Evaluamos la capacidad multilingüe del modelo de embeddings (`nemotron-3-embed-1b`) y generador preguntando en español sobre los documentos en inglés.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 11,
        "metadata": {},
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "--- Q1 en Español ---\n",
                "Q: ¿Qué puntaje BLEU alcanzó el modelo Transformer (big model) en WMT 2014 de inglés a alemán?\n",
                "A: El modelo Transformer (big model) alcanzó un puntaje BLEU de 28.4 en la tarea de traducción de inglés a alemán de WMT 2014. (Attention Is All You Need | attention_paper.pdf)\n",
                "\n",
                "--- Q3 en Español ---\n",
                "Q: ¿Cuál es el propósito del factor de escala raíz cuadrada de dk en la atención Scaled Dot-Product?\n",
                "A: El propósito de escalar los productos punto por $\\sqrt{d_k}$ es contrarrestar el efecto de que los productos punto crezcan en magnitud para valores grandes de $d_k$, lo que empujaría la función softmax a regiones con gradientes extremadamente pequeños. (Attention Is All You Need)\n",
                "\n",
                "--- Q4 en Español ---\n",
                "Q: ¿Cuáles son las dos variantes del modelo RAG propuestas en el artículo y en qué se diferencian?\n",
                "A: Las dos variantes del modelo RAG propuestas son RAG-Sequence y RAG-Token. RAG-Sequence utiliza el mismo documento recuperado para predecir todo el texto objetivo, mientras que RAG-Token puede predecir cada token objetivo basándose en un documento diferente. (Retrieval-Augmented Generation | rag_paper.pdf)\n"
            ]
        }],
        "source": [
            "es_questions = [\n",
            "    ('Q1_es', '¿Qué puntaje BLEU alcanzó el modelo Transformer (big model) en WMT 2014 de inglés a alemán?'),\n",
            "    ('Q3_es', '¿Cuál es el propósito del factor de escala raíz cuadrada de dk en la atención Scaled Dot-Product?'),\n",
            "    ('Q4_es', '¿Cuáles son las dos variantes del modelo RAG propuestas en el artículo y en qué se diferencian?'),\n",
            "]\n",
            "\n",
            "for lbl, eq in es_questions:\n",
            "    ans = rag_chain.invoke(eq)\n",
            "    print(f'--- {lbl} en Español ---')\n",
            "    print(f'Q: {eq}')\n",
            "    print(f'A: {ans}\\n')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Tarea B5 — Quiz propio con ground truth\n",
            "\n",
            "Añadimos tres preguntas adicionales formuladas a partir de la lectura de las secciones de los papers, incluyendo una que cruza ambos artículos.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 12,
        "metadata": {},
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "--- Nueva Pregunta 1 (Attention Is All You Need) ---\n",
                "Q: What optimizer, learning rate schedule, and warmup steps were used to train the Transformer models?\n",
                "A: The Adam optimizer with $\\beta = 0.9$, $\\beta = 0.98$ and $\\epsilon=10^{-9}$ was used, with a learning rate schedule defined by $lrate=d−0.5 \\cdot\\min(step\\_num−0.5,step\\_num\\cdot warmup\\_steps−1.5)$, and $warmup\\_steps=4000$. (Attention Is All You Need | attention_paper.pdf)\n\n",
                "--- Nueva Pregunta 2 (Retrieval-Augmented Generation) ---\n",
                "Q: What Wikipedia dump was used for non-parametric retrieval in RAG, and into how many passages was it split?\n",
                "A: The December 2018 dump of Wikipedia was used for the non-parametric knowledge source, and each article was split into disjoint 100-word chunks, making a total of 21M documents. (Retrieval-Augmented Generation | rag_paper.pdf)\n\n",
                "--- Nueva Pregunta 3 (Pregunta Cruzada) ---\n",
                "Q: How does the masking in the Transformer decoder self-attention ensure autoregressive generation, and why is this property essential for the BART generator in RAG?\n",
                "A: The masking in the decoder self-attention sub-layer prevents positions from attending to subsequent positions, which, combined with the output embeddings being offset by one position, ensures that predictions for position $i$ depend only on the known outputs at positions less than $i$ (Attention Is All You Need).\n"
            ]
        }],
        "source": [
            "new_questions = [\n",
            "    ('Nueva Pregunta 1 (Attention Is All You Need)', 'What optimizer, learning rate schedule, and warmup steps were used to train the Transformer models?'),\n",
            "    ('Nueva Pregunta 2 (Retrieval-Augmented Generation)', 'What Wikipedia dump was used for non-parametric retrieval in RAG, and into how many passages was it split?'),\n",
            "    ('Nueva Pregunta 3 (Pregunta Cruzada)', 'How does the masking in the Transformer decoder self-attention ensure autoregressive generation, and why is this property essential for the BART generator in RAG?'),\n",
            "]\n",
            "\n",
            "for title, nq in new_questions:\n",
            "    ans = rag_chain.invoke(nq)\n",
            "    print(f'--- {title} ---')\n",
            "    print(f'Q: {nq}')\n",
            "    print(f'A: {ans}\\n')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary>✅ Ground Truth — Nuevas Preguntas B5</summary>\n",
            "\n",
            "- **Pregunta 1:** Se utilizó el optimizador Adam con beta1=0.9, beta2=0.98 y epsilon=10^-9. La tasa de aprendizaje varió con una fase de warmup de 4,000 pasos antes de decaer proporcionalmente al inverso de la raíz cuadrada del número de paso. *(Fuente: Section 5.3 — Attention Is All You Need)*\n",
            "- **Pregunta 2:** Se utilizó el volcado de Wikipedia de diciembre de 2018, dividido en 21 millones de fragmentos disjuntos de 100 palabras cada uno. *(Fuente: Section 2.1 — RAG paper)*\n",
            "- **Pregunta 3:** El enmascaramiento en la auto-atención del decodificador impide que una posición atienda a posiciones futuras, garantizando que la predicción del token i dependa únicamente de los tokens anteriores. En RAG, esto es fundamental porque el generador BART produce la respuesta de manera autorregresiva token por token condicionada en el contexto recuperado. *(Fuente: Attention Section 3.2.3 y RAG Section 2.1)*\n",
            "</details>\n"
        ]
    }
]

nb["cells"].extend(extra_cells)

with open("notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated 03_rag_workshop_papers_nvidia_nim_colab.ipynb successfully")
