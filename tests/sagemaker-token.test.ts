import { expect, test } from "bun:test";
import { createSageMakerBearerToken } from "../src/embeddings/sagemaker-token.ts";

test("creates the SageMaker short-lived bearer token envelope without a network call", async () => {
  const token = await createSageMakerBearerToken({
    region: "eu-west-2",
    expiresIn: 300,
    credentials: {
      accessKeyId: "AKIDEXAMPLE",
      secretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
      sessionToken: "temporary-session-token"
    }
  });

  expect(token.startsWith("sagemaker-api-key-")).toBe(true);
  const decoded = Buffer.from(token.slice("sagemaker-api-key-".length), "base64").toString("utf8");
  expect(decoded.startsWith("sagemaker.amazonaws.com/?")).toBe(true);
  expect(decoded).toContain("Action=CallWithBearerToken");
  expect(decoded).toContain("X-Amz-Algorithm=AWS4-HMAC-SHA256");
  expect(decoded).toContain("X-Amz-Expires=300");
  expect(decoded).toContain("X-Amz-Signature=");
  expect(decoded).toContain("Version=1");
  expect(decoded).not.toContain("EXAMPLEKEY");
});
