import type { PredictionInputs } from "@/contexts/HistoryContext";

const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "";

export interface PredictResponse {
  predicted_vehicles: number;
  model?: string;
}

export async function predictTraffic(inputs: PredictionInputs): Promise<PredictResponse> {
  const params = new URLSearchParams({
    junction: String(inputs.junction),
    day_of_week: String(inputs.day_of_week),
    hour: String(inputs.hour),
    month: String(inputs.month),
    year: String(inputs.year),
  });

  if (!API_URL) {
    // Graceful fallback — deterministic mock so the UI is always usable
    await new Promise((r) => setTimeout(r, 800));
    const base = 50 + inputs.junction * 30;
    const rush = inputs.hour >= 7 && inputs.hour <= 9 ? 80 : inputs.hour >= 16 && inputs.hour <= 19 ? 95 : 10;
    const weekend = inputs.day_of_week >= 5 ? -25 : 0;
    return {
      predicted_vehicles: Math.max(0, Math.round(base + rush + weekend + Math.random() * 20)),
      model: "TrafficNet-v1 (mock)",
    };
  }

  const res = await fetch(`${API_URL}/predict?${params.toString()}`, { method: "POST" });
  if (!res.ok) throw new Error(`Prediction failed (${res.status})`);
  const data = await res.json();
  return {
    predicted_vehicles: Number(data.predicted_vehicles ?? data.prediction ?? 0),
    model: data.model ?? "TrafficNet",
  };
}
