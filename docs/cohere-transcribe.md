# Cohere Transcribe Integration Notes

TurboSearch will use Cohere Transcribe 03-2026 as the local/production transcription model for audio and video evidence.

Current status: Cohere API provider is wired for local testing; vLLM-compatible provider is wired for a private endpoint with the same multipart response shape.

Reference:

- Hugging Face model: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/tree/main

Known model facts from the model card:

- Task: automatic speech recognition
- Model size: 2B parameters
- Architecture: Conformer encoder-decoder
- License: Apache-2.0
- Supports 14 languages
- Files are gated; users must accept the Hugging Face conditions before downloading weights
- Supported runtimes listed by Cohere include Transformers, vLLM, MLX Audio, Rust, browser/WebGPU, and related ecosystem tools

## Ollama

I do not see Ollama support listed for this model. Ollama is not the right first target for Cohere Transcribe.

## Runtime Plan

Recommended split:

- local testing: Cohere API
- private local development: Transformers or MLX Audio service, depending on Apple Silicon performance
- production: vLLM server or Cohere Model Vault

The model card recommends Transformers for offline inference and vLLM for online/production serving.

vLLM serving shape from the model card:

```bash
vllm serve CohereLabs/cohere-transcribe-03-2026 --trust-remote-code
```

Request shape:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/absolute/path/audio.wav" \
  -F "model=CohereLabs/cohere-transcribe-03-2026"
```

TurboSearch hosted API config:

```bash
TURBOSEARCH_TRANSCRIPTION_PROVIDER=cohere-api
COHERE_API_KEY=...
COHERE_TRANSCRIBE_MODEL=cohere-transcribe-03-2026
COHERE_TRANSCRIBE_LANGUAGE=en
```

vLLM/private endpoint config:

```bash
TURBOSEARCH_TRANSCRIPTION_PROVIDER=vllm-cohere-transcribe
COHERE_TRANSCRIBE_ENDPOINT=http://127.0.0.1:8000/v1/audio/transcriptions
COHERE_TRANSCRIBE_MODEL=CohereLabs/cohere-transcribe-03-2026
```

Video files are converted to mono 16k WAV with `ffmpeg` before transcription. Audio files are sent directly.

## Timestamp Support

Cohere's current hosted Transcribe API returns transcript text. Cohere's model documentation says timestamps and speaker diarization are not supported by this model.

TurboSearch stores Cohere API output as an untimestamped transcript block. The internal transcription contract supports timestamped `segments`, but only providers that return real segment timestamps should populate them. TurboSearch intentionally does not infer transcript timestamps from text position.
