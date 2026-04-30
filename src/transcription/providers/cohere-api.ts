import type {
  TranscriptionProvider,
  TranscriptionRequest,
  TranscriptionResponse
} from "../provider.ts";

type CohereTranscriptionOptions = {
  apiKey: string;
  endpoint: string;
  model: string;
  language: string;
};

type CohereTranscriptionResponse = {
  text?: string;
  language?: string;
};

export class CohereApiTranscriptionProvider implements TranscriptionProvider {
  readonly name = "cohere-api" as const;
  private readonly apiKey: string;
  private readonly endpoint: string;
  private readonly model: string;
  private readonly language: string;

  constructor(options: CohereTranscriptionOptions) {
    this.apiKey = options.apiKey;
    this.endpoint = options.endpoint;
    this.model = options.model;
    this.language = options.language;
  }

  async transcribe(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    if (!this.apiKey) {
      throw new Error("COHERE_API_KEY is required for TURBOSEARCH_TRANSCRIPTION_PROVIDER=cohere-api.");
    }

    const form = new FormData();
    form.append("model", this.model);
    form.append("language", request.language ?? this.language);
    form.append("file", Bun.file(request.audioPath));

    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`
      },
      body: form
    });

    if (!response.ok) {
      throw new Error(`Cohere transcription request failed: ${response.status} ${await response.text()}`);
    }

    const json = await response.json() as CohereTranscriptionResponse;
    const text = (json.text ?? "").trim();
    if (!text) throw new Error("Cohere transcription returned empty text.");

    return {
      text,
      model: this.model,
      provider: this.name,
      language: json.language ?? request.language ?? this.language
    };
  }
}
