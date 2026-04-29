# GLM-OCR Integration Notes

TurboSearch will use Z.ai GLM-OCR as the local OCR model for image evidence and video frame OCR.

Current status: Ollama local provider wired for images; vLLM production provider reserved but not wired yet.

References:

- Hugging Face model: https://huggingface.co/zai-org/GLM-OCR
- Z.ai GLM-OCR guide: https://docs.z.ai/guides/vlm/glm-ocr

Known model facts from the docs:

- Task: image-to-text / OCR
- Model size: 0.9B parameters
- License: MIT for the model
- Supports images and PDFs; Z.ai docs list JPG/PNG and PDF inputs
- Supports text, Markdown documents, and image links as output modalities
- Supports multiple languages, including Chinese, English, French, Spanish, Russian, German, Japanese, and Korean
- Deployment options include the official SDK, vLLM, SGLang, Ollama, and Transformers

## Runtime Plan

TurboSearch uses a provider boundary:

- local development: Ollama with `glm-ocr:latest`
- production: vLLM GLM-OCR service, pending exact deployment contract

The code calls Ollama directly from Bun for local development. It uses Ollama's native API rather than an OpenAI-compatible endpoint because the upstream GLM-OCR Ollama guide recommends native `/api/generate` for vision requests.

Local setup:

```bash
ollama pull glm-ocr:latest
ollama serve
```

TurboSearch env:

```bash
TURBOSEARCH_OCR_PROVIDER=ollama-glm-ocr
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_GLM_OCR_MODEL=glm-ocr:latest
TURBOSEARCH_OCR_PROMPT=Text Recognition:
```

The Bun adapter sends:

```http
POST /api/generate
Content-Type: application/json
```

Request:

```json
{
  "model": "glm-ocr:latest",
  "prompt": "Text Recognition:",
  "images": ["base64-image"],
  "stream": false,
  "options": { "temperature": 0 }
}
```

Response:

```json
{
  "response": "Extracted text..."
}
```

TurboSearch indexes the response text as `image_ocr` evidence. If Ollama is offline or returns no text, extraction emits a pending OCR event and falls back to sidecar text if present.

## Prompt Modes

The model card lists document parsing prompts such as:

- `Text Recognition:`
- `Formula Recognition:`
- `Table Recognition:`

For the first integration, use `Text Recognition:` as the default. Add table/formula routing later if extraction quality demands it.

## Production vLLM Plan

Keep the TypeScript interface stable and add a `vllm-glm-ocr` provider once the production server contract is chosen.

Expected env:

```bash
TURBOSEARCH_OCR_PROVIDER=vllm-glm-ocr
GLM_OCR_ENDPOINT=http://internal-ocr:8000
```

Open production decision: whether vLLM should expose an OpenAI-compatible vision endpoint or a small custom `/ocr` service that normalizes responses into text, Markdown, and optional blocks.
