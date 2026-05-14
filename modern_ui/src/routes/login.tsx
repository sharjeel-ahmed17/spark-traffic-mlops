import { createFileRoute, useRouter, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useState, type FormEvent } from "react";
import { TrafficCone, Mail, Lock, Loader2, AlertCircle } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export const Route = createFileRoute("/login")({
  component: LoginPage,
});

function LoginPage() {
  const { login, isAuthenticated } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("user@example.com");
  const [password, setPassword] = useState("123456");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (isAuthenticated) {
    router.navigate({ to: "/dashboard" });
  }

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      router.navigate({ to: "/dashboard" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-md glass-card rounded-3xl p-8 shadow-2xl shadow-primary/10"
      >
        <div className="flex flex-col items-center text-center mb-8">
          <motion.div
            initial={{ rotate: -20, scale: 0 }}
            animate={{ rotate: 0, scale: 1 }}
            transition={{ type: "spring", delay: 0.15 }}
            className="size-16 rounded-2xl gradient-primary flex items-center justify-center shadow-xl shadow-primary/40 mb-4"
          >
            <TrafficCone className="size-8 text-primary-foreground" />
          </motion.div>
          <h1 className="text-3xl font-bold text-gradient">TrafficVP</h1>
          <p className="text-sm text-muted-foreground mt-1">Sign in to predict vehicle traffic</p>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-1.5 block">Email</label>
            <div className="relative">
              <Mail className="size-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-background border border-input focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition"
              />
            </div>
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">Password</label>
            <div className="relative">
              <Lock className="size-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full pl-10 pr-3 py-2.5 rounded-xl bg-background border border-input focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition"
              />
            </div>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded-lg p-3"
            >
              <AlertCircle className="size-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full gradient-primary text-primary-foreground font-semibold rounded-xl py-3 shadow-lg shadow-primary/30 hover:shadow-primary/50 transition-shadow flex items-center justify-center gap-2 disabled:opacity-70"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : null}
            {loading ? "Signing in..." : "Sign in"}
          </button>

          <p className="text-xs text-center text-muted-foreground pt-2">
            Demo: <code className="bg-muted px-1 py-0.5 rounded">user@example.com</code> /{" "}
            <code className="bg-muted px-1 py-0.5 rounded">123456</code>
          </p>
          <p className="text-xs text-center text-muted-foreground">
            <Link to="/dashboard" className="text-primary hover:underline">Back to app</Link>
          </p>
        </form>
      </motion.div>
    </div>
  );
}
