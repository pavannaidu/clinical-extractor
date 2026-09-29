export const MAX_OUTPUT_TOKENS_CEILING = 8192;

export const OUTPUT_TOKEN_OVERRIDE_ERROR =
  `Enter a whole number from 1 to ${MAX_OUTPUT_TOKENS_CEILING.toLocaleString()}, or leave blank.`;

export interface OutputTokenOverrideValidation {
  value: number | null;
  error: string | null;
}

export function validateOutputTokenOverride(
  input: string
): OutputTokenOverrideValidation {
  const trimmed = input.trim();
  if (!trimmed) {
    return { value: null, error: null };
  }

  if (!/^\d+$/.test(trimmed)) {
    return { value: null, error: OUTPUT_TOKEN_OVERRIDE_ERROR };
  }

  const value = Number(trimmed);
  if (
    !Number.isSafeInteger(value) ||
    value < 1 ||
    value > MAX_OUTPUT_TOKENS_CEILING
  ) {
    return { value: null, error: OUTPUT_TOKEN_OVERRIDE_ERROR };
  }

  return { value, error: null };
}
