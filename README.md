## LLM + RAG Clothing Recommendation System: Full Details

---

## 1. Vision-Based Feature Extraction (LLM Stage 1)

This stage uses a large multimodal model (in this case: **GPT-5-nano** through the OpenAI API) to perform image-to-structured-data conversion.

### A. Image Preprocessing

The user's input image (specified by `IMAGE_PATH`) is loaded and converted into a **Base64 string**. This is the standard method for transmitting binary image data within an API request payload. The function `get_llm_prediction_for_image` sends the Base64 image and the instruction prompt to the OpenAI API.

### B. The Normalization Prompt (`PROMPT_TEXT_v2`)

This is the core of the extraction. The LLM acts as a **"fashion normalizer"** and is strictly instructed to output a JSON object based on a predefined schema. The fields extracted are essential for the downstream search process:

| JSON Field | Purpose | Significance |
| :--- | :--- | :--- |
| `category`, `article_type` | High-level categorization. | Used for **initial hard-filtering** of the RAG knowledge base. |
| `color_primary`, `color_primary_simple` | Detailed and simplified color. | Used for **initial hard-filtering** to narrow the search space. |
| `product_display_name` | A natural language description of the item. | Becomes the **text query** used for the semantic search (KNN). |
| `material`, `details` | Granular descriptive features. | **Appended to the text query** to enrich the semantic search. |

---

## 2. Retrieval-Augmented Generation (RAG) & KNN Search

This stage uses the structured output from the LLM to efficiently search a large database for highly relevant clothing items.

### A. Knowledge Base Setup (`setup_knowledge_base_for_rag`)

1.  A knowledge base (KB) is created from a `styles.csv` file, filtered to include only 'Apparel'.
2.  The **`productDisplayName`** of every item in the KB is converted into a **vector embedding** using a **Sentence Transformer model** (`all-mpnet-base-v2`).
3.  This KB (containing clothing data and its corresponding embeddings) is saved as a pickle file for caching, allowing for fast loading on subsequent runs.

### B. Attribute Filtering (`filter_primary_attributes`)

This is the crucial **"Retrieval"** part of the RAG system. The structured attributes (`category`, `article_type`, `baseColor`, `baseColorSimple`) extracted by the LLM are used to perform **strict, deterministic filtering** on the KB (DataFrame). This dramatically reduces the number of items that need to be searched semantically.

### C. Semantic Search (KNN)

1.  A composite **text query** is generated from the LLM's output by combining the `product_display_name` and the list of `details`.
2.  This query is converted into a **query embedding** using the same Sentence Transformer model.
3.  **K-Nearest Neighbors (KNN)** is applied to find the 10 closest item embeddings in the *filtered* KB to the query embedding. These 10 items represent the top semantic matches based on text description.

---

## 3. Final LLM Re-ranking (LLM Stage 2)

The 10 items retrieved by the RAG/KNN process are refined by sending them back to the multimodal LLM for a final, visual similarity check.

### A. Image Collection and Preprocessing

* The images for the 10 candidate items are fetched via their URLs.
* The function `resize_base64_image` resizes all fetched images to a smaller size (e.g., **200x200 pixels**) to manage the token/context limits of the multimodal LLM API and reduce latency.
* All images are encoded to Base64.

### B. The Selection Prompt (`PROMPT_FINAL_v1`)

* The LLM is given two sets of images: the **original model image** and the **10 resized candidate item images**.
* The prompt instructs the LLM to act as a **"fashion expert"** and **visually compare** the candidate items against the clothing worn by the model.
* The output must be a JSON list where items are sorted by **`similarity_order`** (1 being the most similar).

### C. Final Output

* The final response is parsed, and the items are sorted by the LLM's `similarity_order`.
* The top 5 (or a specified number) most similar items are selected, and their images are displayed, representing the final, context-aware, and visually validated recommendation.
