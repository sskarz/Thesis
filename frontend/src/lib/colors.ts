const KNOWN_COLORS: Record<string, string> = {
  'qwen2.5-3b': '#2563eb',
  'llama-3.2-3b': '#dc2626',
  'qwen3.5-4b': '#059669',
  'functiongemma-270m': '#d97706',
};

const PALETTE = [
  '#7c3aed', '#0891b2', '#be185d', '#65a30d',
  '#ea580c', '#4f46e5', '#0d9488', '#c026d3',
];

const assignedColors: Record<string, string> = {};
let nextIdx = 0;

export function modelColor(modelKey: string): string {
  if (KNOWN_COLORS[modelKey]) return KNOWN_COLORS[modelKey];
  if (!assignedColors[modelKey]) {
    assignedColors[modelKey] = PALETTE[nextIdx % PALETTE.length];
    nextIdx++;
  }
  return assignedColors[modelKey];
}

export function hexAlpha(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}
