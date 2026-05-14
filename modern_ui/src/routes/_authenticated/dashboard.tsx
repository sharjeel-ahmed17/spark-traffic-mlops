import { createFileRoute } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Car, Calendar, Clock, MapPin, Sparkles, Loader2, AlertCircle, Gauge } from "lucide-react";
import { predictTraffic } from "@/lib/predict";
import { useHistory, type PredictionInputs } from "@/contexts/HistoryContext";

export const Route = createFileRoute("/_authenticated/dashboard")({
  component: Dashboard,
});

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

interface Result {
  predicted_vehicles: number;
  model: string;
  inputs: PredictionInputs;
}

function Dashboard() {
  const { add } = useHistory();
  const [inputs, setInputs] = useState<PredictionInputs>({
    junction: 1,
    day_of_week: 0,
    hour: 8,
    month: new Date().getMonth() + 1,
    year: new Date().getFullYear(),
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);

  const update = <K extends keyof PredictionInputs>(k: K, v: PredictionInputs[K]) =>
    setInputs((s) => ({ ...s, [k]: v }));

  const onPredict = async () => {
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await predictTraffic(inputs);
      const model = res.model || "TrafficNet";
      const rec = add({ inputs, predicted_vehicles: res.predicted_vehicles, model });
      setResult({ predicted_vehicles: rec.predicted_vehicles, model, inputs });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:py-12">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass-card text-xs font-medium mb-3">
          <Sparkles className="size-3 text-primary" /> AI Traffic Forecast
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight">
          Predict <span className="text-gradient">Vehicle Traffic</span>
        </h1>
        <p className="text-muted-foreground mt-2 max-w-xl mx-auto">
          Enter junction and time details to forecast vehicle volume.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card rounded-3xl p-6 sm:p-8 shadow-xl shadow-primary/10"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {/* Junction */}
          <Field icon={<MapPin className="size-4" />} label="Junction">
            <select
              value={inputs.junction}
              onChange={(e) => update("junction", Number(e.target.value))}
              className="w-full px-3 py-2.5 rounded-xl bg-background border border-input focus:border-primary outline-none"
            >
              {[1, 2, 3, 4].map((j) => (
                <option key={j} value={j}>Junction {j}</option>
              ))}
            </select>
          </Field>

          {/* Day */}
          <Field icon={<Calendar className="size-4" />} label="Day of Week">
            <select
              value={inputs.day_of_week}
              onChange={(e) => update("day_of_week", Number(e.target.value))}
              className="w-full px-3 py-2.5 rounded-xl bg-background border border-input focus:border-primary outline-none"
            >
              {DAYS.map((d, i) => (
                <option key={d} value={i}>{d}</option>
              ))}
            </select>
          </Field>

          {/* Hour */}
          <Field icon={<Clock className="size-4" />} label={`Hour: ${String(inputs.hour).padStart(2, "0")}:00`}>
            <input
              type="range"
              min={0}
              max={23}
              value={inputs.hour}
              onChange={(e) => update("hour", Number(e.target.value))}
              className="w-full accent-[var(--primary)]"
            />
          </Field>

          {/* Month */}
          <Field icon={<Calendar className="size-4" />} label={`Month: ${MONTHS[inputs.month - 1]}`}>
            <input
              type="range"
              min={1}
              max={12}
              value={inputs.month}
              onChange={(e) => update("month", Number(e.target.value))}
              className="w-full accent-[var(--primary)]"
            />
          </Field>

          {/* Year */}
          <Field icon={<Calendar className="size-4" />} label="Year">
            <input
              type="number"
              min={2020}
              max={2030}
              value={inputs.year}
              onChange={(e) => update("year", Math.min(2030, Math.max(2020, Number(e.target.value))))}
              className="w-full px-3 py-2.5 rounded-xl bg-background border border-input focus:border-primary outline-none"
            />
          </Field>
        </div>

        <button
          onClick={onPredict}
          disabled={loading}
          className="mt-8 w-full gradient-primary text-primary-foreground font-semibold rounded-2xl py-4 text-lg shadow-xl shadow-primary/40 hover:shadow-primary/60 transition-shadow flex items-center justify-center gap-2 disabled:opacity-70"
        >
          {loading ? (
            <>
              <Loader2 className="size-5 animate-spin" /> Predicting...
            </>
          ) : (
            <>🔮 Predict</>
          )}
        </button>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-4 flex items-start gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-xl p-3"
            >
              <AlertCircle className="size-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ type: "spring", stiffness: 120, damping: 16 }}
            className="mt-6 glass-card rounded-3xl p-6 sm:p-8 shadow-xl"
          >
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Gauge className="size-4 text-primary" /> Predicted Vehicles
            </div>
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.1 }}
              className="text-6xl sm:text-7xl font-bold text-gradient mt-2 flex items-baseline gap-3"
            >
              {result.predicted_vehicles}
              <Car className="size-8 text-primary" />
            </motion.div>
            <div className="text-xs uppercase tracking-widest text-muted-foreground mt-3">
              Model: <span className="text-foreground font-medium">{result.model}</span>
            </div>
            <div className="mt-5 grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
              <Stat label="Junction" value={result.inputs.junction} />
              <Stat label="Day" value={DAYS[result.inputs.day_of_week]} />
              <Stat label="Hour" value={`${String(result.inputs.hour).padStart(2, "0")}:00`} />
              <Stat label="Month" value={MONTHS[result.inputs.month - 1]} />
              <Stat label="Year" value={result.inputs.year} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Field({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-sm font-medium mb-1.5 flex items-center gap-1.5 text-foreground/90">
        <span className="text-primary">{icon}</span> {label}
      </label>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-muted/50 border border-border/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="font-semibold mt-0.5 truncate">{value}</div>
    </div>
  );
}
