import { Link, useRouter } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Car, History, LogOut, Moon, Sun, TrafficCone } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useTheme } from "@/contexts/ThemeContext";

export function AppShell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    router.navigate({ to: "/login" });
  };

  return (
    <div className="min-h-screen flex flex-col">
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="sticky top-0 z-40 glass-card border-b"
      >
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <Link to="/dashboard" className="flex items-center gap-2 group">
            <div className="size-9 rounded-xl gradient-primary flex items-center justify-center shadow-lg shadow-primary/30">
              <TrafficCone className="size-5 text-primary-foreground" />
            </div>
            <div className="leading-tight">
              <div className="font-bold text-gradient text-lg">TrafficVP</div>
              <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Vehicle Prediction</div>
            </div>
          </Link>

          <nav className="flex items-center gap-1 sm:gap-2">
            <Link
              to="/dashboard"
              className="px-3 py-2 rounded-lg text-sm font-medium hover:bg-accent transition-colors flex items-center gap-1.5"
              activeProps={{ className: "bg-accent text-accent-foreground" }}
            >
              <Car className="size-4" /> <span className="hidden sm:inline">Predict</span>
            </Link>
            <Link
              to="/history"
              className="px-3 py-2 rounded-lg text-sm font-medium hover:bg-accent transition-colors flex items-center gap-1.5"
              activeProps={{ className: "bg-accent text-accent-foreground" }}
            >
              <History className="size-4" /> <span className="hidden sm:inline">History</span>
            </Link>
            <button
              onClick={toggle}
              aria-label="Toggle theme"
              className="size-9 rounded-lg hover:bg-accent flex items-center justify-center transition-colors"
            >
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
            <button
              onClick={handleLogout}
              className="size-9 rounded-lg hover:bg-destructive/10 hover:text-destructive flex items-center justify-center transition-colors"
              aria-label="Logout"
              title={user?.email}
            >
              <LogOut className="size-4" />
            </button>
          </nav>
        </div>
      </motion.header>
      <main className="flex-1">{children}</main>
      <footer className="text-center text-xs text-muted-foreground py-6">
        Built with TrafficNet · {user?.email}
      </footer>
    </div>
  );
}
