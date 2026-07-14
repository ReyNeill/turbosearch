import { Sha256 } from "@aws-crypto/sha256-js";
import { defaultProvider } from "@aws-sdk/credential-provider-node";
import { HttpRequest } from "@smithy/protocol-http";
import { SignatureV4 } from "@smithy/signature-v4";

export type SageMakerTokenOptions = {
  region: string;
  profile?: string;
  expiresIn?: number;
  credentials?: {
    accessKeyId: string;
    secretAccessKey: string;
    sessionToken?: string;
  };
};

export async function createSageMakerBearerToken(options: SageMakerTokenOptions): Promise<string> {
  const expiresIn = options.expiresIn ?? 300;
  if (!Number.isInteger(expiresIn) || expiresIn < 1 || expiresIn > 43_200) {
    throw new Error("SageMaker bearer token expiry must be between 1 and 43200 seconds.");
  }

  const credentials = options.credentials
    ?? defaultProvider(options.profile ? { profile: options.profile } : {});
  const signer = new SignatureV4({
    credentials,
    region: options.region,
    service: "sagemaker",
    sha256: Sha256
  });
  const request = new HttpRequest({
    protocol: "https:",
    hostname: "sagemaker.amazonaws.com",
    method: "POST",
    path: "/",
    headers: { host: "sagemaker.amazonaws.com" },
    query: { Action: "CallWithBearerToken" }
  });
  const signed = await signer.presign(request, { expiresIn });
  const query = Object.entries(signed.query ?? {})
    .flatMap(([key, value]) => Array.isArray(value) ? value.map((item) => [key, item] as const) : [[key, value] as const])
    .filter((entry): entry is readonly [string, string] => typeof entry[1] === "string")
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
    .join("&");
  const presignedUrl = `sagemaker.amazonaws.com/?${query}&Version=1`;
  return `sagemaker-api-key-${Buffer.from(presignedUrl, "utf8").toString("base64")}`;
}
