const SUPPORTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx"] as const;
const BINARY_EXTENSIONS = [".pdf", ".docx"] as const;

function getExtension(file: File): string {
  const dotIndex = file.name.lastIndexOf(".");
  return dotIndex >= 0 ? file.name.slice(dotIndex).toLowerCase() : "";
}

export function isSupportedUploadFile(file: File): boolean {
  return SUPPORTED_EXTENSIONS.includes(getExtension(file) as (typeof SUPPORTED_EXTENSIONS)[number]);
}

export function isBinaryUploadFile(file: File): boolean {
  return BINARY_EXTENSIONS.includes(getExtension(file) as (typeof BINARY_EXTENSIONS)[number]);
}
