import { createContext, useContext, useEffect, useState, ReactNode } from "react";

export interface PredictionInputs {
  junction: number;
  day_of_week: number; // 0=Mon..6=Sun
  hour: number;
  month: number;
  year: number;
}

export interface PredictionRecord {
  id: string;
  timestamp: number;
  inputs: PredictionInputs;
  predicted_vehicles: number;
  model: string;
}

interface HistoryContextValue {
  history: PredictionRecord[];
  add: (rec: Omit<PredictionRecord, "id" | "timestamp">) => PredictionRecord;
  clear: () => void;
}

const HistoryContext = createContext<HistoryContextValue | undefined>(undefined);
const STORAGE_KEY = "tvp_history";

export function HistoryProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<PredictionRecord[]>([]);

  useEffect(() => {
    try {
      const raw = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
      if (raw) setHistory(JSON.parse(raw));
    } catch {}
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  const add: HistoryContextValue["add"] = (rec) => {
    const full: PredictionRecord = { ...rec, id: crypto.randomUUID(), timestamp: Date.now() };
    setHistory((h) => [full, ...h].slice(0, 100));
    return full;
  };

  const clear = () => setHistory([]);

  return <HistoryContext.Provider value={{ history, add, clear }}>{children}</HistoryContext.Provider>;
}

export function useHistory() {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error("useHistory must be used within HistoryProvider");
  return ctx;
}
