import type { AppConfig } from "../config.ts";
import type { TranscriptionProviderName } from "../types.ts";
import { CohereApiTranscriptionProvider } from "./providers/cohere-api.ts";
import { VllmCohereTranscriptionProvider } from "./providers/vllm.ts";

export type TranscriptionRequest = {
  audioPath: string;
  language?: string;
};

export type TranscriptionResponse = {
  text: string;
  model: string;
  provider: TranscriptionProviderName;
  language?: string;
  segments?: TranscriptionSegment[];
};

export type TranscriptionSegment = {
  text: string;
  startMs: number;
  endMs: number;
};

export interface TranscriptionProvider {
  readonly name: TranscriptionProviderName;
  transcribe(request: TranscriptionRequest): Promise<TranscriptionResponse>;
}

export function createTranscriptionProvider(config: AppConfig): TranscriptionProvider | null {
  if (config.transcriptionProvider === "none") return null;

  if (config.transcriptionProvider === "vllm-cohere-transcribe") {
    return new VllmCohereTranscriptionProvider({
      endpoint: config.cohereTranscribeEndpoint,
      model: config.cohereTranscribeModel,
      language: config.cohereTranscribeLanguage
    });
  }

  return new CohereApiTranscriptionProvider({
    apiKey: config.cohereApiKey,
    endpoint: config.cohereTranscribeEndpoint,
    model: config.cohereTranscribeModel,
    language: config.cohereTranscribeLanguage
  });
}
