import type {
  TranscriptionProvider,
  TranscriptionRequest,
  TranscriptionResponse
} from "../provider.ts";
import { parseTranscriptionSegments } from "../segments.ts";

type VllmTranscriptionOptions = {
  endpoint: string;
  model: string;
  language: string;
};

type VllmTranscriptionResponse = {
  text?: string;
  language?: string;
  segments?: unknown;
};

export class VllmCohereTranscriptionProvider implements TranscriptionProvider {
  readonly name = "vllm-cohere-transcribe" as const;
  private readonly endpoint: string;
  private readonly model: string;
  private readonly language: string;

  constructor(options: VllmTranscriptionOptions) {
    this.endpoint = options.endpoint;
    this.model = options.model;
    this.language = options.language;
  }

  async transcribe(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    const form = new FormData();
    form.append("model", this.model);
    form.append("language", request.language ?? this.language);
    form.append("file", Bun.file(request.audioPath));

    const response = await fetch(this.endpoint, {
      method: "POST",
      body: form
    });

    if (!response.ok) {
      throw new Error(`vLLM transcription request failed: ${response.status} ${await response.text()}`);
    }

    const json = await response.json() as VllmTranscriptionResponse;
    const text = (json.text ?? "").trim();
    if (!text) throw new Error("vLLM transcription returned empty text.");

    return {
      text,
      model: this.model,
      provider: this.name,
      language: json.language ?? request.language ?? this.language,
      segments: parseTranscriptionSegments(json.segments)
    };
  }
}
