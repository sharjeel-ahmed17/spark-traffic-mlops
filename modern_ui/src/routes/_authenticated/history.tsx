import { createFileRoute } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { Car, Trash2, Inbox } from "lucide-react";
import { useHistory } from "@/contexts/HistoryContext";

export const Route = createFileRoute("/_authenticated/history")({
  component: HistoryPage,
});

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function HistoryPage() {
  const { history, clear } = useHistory();

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:py-12">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold">Prediction History</h1>
          <p className="text-sm text-muted-foreground">{history.length} saved {history.length === 1 ? "prediction" : "predictions"}</p>
        </div>
        {history.length > 0 && (
          <button
            onClick={() => {
              if (confirm("Clear all predictions?")) clear();
            }}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-medium border border-destructive/40 text-destructive hover:bg-destructive/10 transition"
          >
            <Trash2 className="size-4" /> Clear
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-3xl p-12 text-center"
        >
          <Inbox className="size-12 mx-auto text-muted-foreground mb-3" />
          <p className="font-medium">No predictions yet</p>
          <p className="text-sm text-muted-foreground mt-1">Run your first prediction from the dashboard.</p>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <AnimatePresence>
            {history.map((rec, i) => (
              <motion.div
                key={rec.id}
                layout
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ delay: Math.min(i * 0.04, 0.3) }}
                className="glass-card rounded-2xl p-5 hover:shadow-lg hover:shadow-primary/10 transition-shadow"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(rec.timestamp).toLocaleString()}
                    </div>
                    <div className="text-3xl font-bold text-gradient mt-1 flex items-center gap-2">
                      {rec.predicted_vehicles}
                      <Car className="size-5 text-primary" />
                    </div>
                  </div>
                  <span className="text-[10px] uppercase tracking-widest px-2 py-1 rounded-full bg-primary/10 text-primary">
                    {rec.model}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 text-xs">
                  <Tag>J{rec.inputs.junction}</Tag>
                  <Tag>{DAYS[rec.inputs.day_of_week]}</Tag>
                  <Tag>{String(rec.inputs.hour).padStart(2, "0")}:00</Tag>
                  <Tag>{MONTHS[rec.inputs.month - 1]} {rec.inputs.year}</Tag>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return <span className="px-2 py-0.5 rounded-md bg-muted border border-border/50">{children}</span>;
}
