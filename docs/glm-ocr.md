# GLM-OCR Integration Notes

TurboSearch will use Z.ai GLM-OCR as the local OCR model for image evidence and video frame OCR.

Current status: Ollama local provider is wired for images, scanned PDF pages, and video frames. The vLLM provider is wired against an OpenAI-compatible `/v1/chat/completions` vision endpoint for production deployments.

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
- production: vLLM GLM-OCR service through an OpenAI-compatible vision endpoint

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
TURBOSEARCH_OCR_MODE=text
TURBOSEARCH_OCR_PROMPT=
TURBOSEARCH_OCR_CONTEXT_TOKENS=8192
TURBOSEARCH_OCR_MAX_OUTPUT_TOKENS=2048
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
  "options": {
    "temperature": 0,
    "num_ctx": 8192,
    "num_predict": 2048
  }
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

TurboSearch now exposes these modes explicitly:

- `text` -> `Text Recognition:`
- `table` -> `Table Recognition:`
- `formula` -> `Formula Recognition:`
- `figure` -> `Figure Recognition:`

Set `TURBOSEARCH_OCR_MODE=table` or pass JSON like `{"ocrMode":"table"}` for known table-heavy inputs. A custom `ocrPrompt` overrides the mode prompt.

## Production vLLM Plan

The `vllm-glm-ocr` provider sends an OpenAI-compatible chat completion request with a text OCR prompt and a base64 data URL image.

Expected env:

```bash
TURBOSEARCH_OCR_PROVIDER=vllm-glm-ocr
VLLM_GLM_OCR_BASE_URL=http://internal-ocr:8000/v1
VLLM_GLM_OCR_API_KEY=
VLLM_GLM_OCR_MODEL=zai-org/GLM-OCR
```

If the production deployment uses a custom `/ocr` endpoint instead, keep the provider interface stable and adapt only `src/ocr/providers/vllm.ts`.
