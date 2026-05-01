import customtkinter as ctk
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
from typing import Dict

# --- 1. CONFIGURATION & CONSTANTS ---
# Moving settings to a central place makes the code "Clean"
REFRESH_MS = 100
HISTORY_LEN = 100
THEME_COLOR = "#32FF6A"
ALERT_COLOR = "#FF3232"

@dataclass
class BioState:
    """Stores the parameters for a specific biological scenario."""
    s: float  # Signal Strength
    n: float  # Hill Coefficient
    d: float  # Degradation Rate
    o2: float # Oxygen Level
    color: str

SCENARIOS: Dict[str, BioState] = {
    "Homeostasis":     BioState(0.15, 2.0, 0.50, 1.0, "#1f77b4"),
    "Stressed":        BioState(0.20, 3.0, 0.12, 1.0, "#ff7f0e"),
    "Overexpression":  BioState(0.24, 3.0, 0.09, 0.9, "#d62728"),
    "Inhibitors":      BioState(0.10, 2.0, 0.06, 1.0, "#2ca02c"),
    "PROTAC":          BioState(0.10, 2.0, 0.20, 1.0, "#2ca02c"),
}

# --- 2. THE MODEL (The Biophysics Engine) ---
class BiophysicsEngine:
    """Handles the mathematical simulation independently of the UI."""
    def __init__(self, k_threshold: float = 0.5):
        self.k = k_threshold
        self.history = [0.1] * HISTORY_LEN
        
    def hill_equation(self, x, t, s, n, o2, d):
        """Mathematical heart: dX/dt = Production - Degradation"""
        production = o2 * (np.power(s, n) / (np.power(self.k, n) + np.power(s, n)))
        return production - (d * x)

    def compute_next_step(self, state: BioState) -> float:
        """Solves the ODE for the next time increment."""
        t_span = np.linspace(0, 0.5, 7)
        sol = odeint(self.hill_equation, self.history[-1], t_span, 
                     args=(state.s, state.n, state.o2, state.d))
        
        # Add stochastic noise
        new_val = max(0, sol[-1][0] + np.random.normal(0, 0.01))
        self.history.append(new_val)
        self.history.pop(0)
        return new_val

# --- 3. THE VIEW (The CustomTkinter GUI) ---
class CellFateApp(ctk.CTk):
    def __init__(self, engine: BiophysicsEngine):
        super().__init__()
        self.engine = engine
        self.current_mode = "Homeostasis"
        
        # UI Setup
        self._setup_window()
        self._create_sidebar()
        self._setup_plot()
        
        # Start Loop
        self.run_engine_loop()

    def _setup_window(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.title("Cell Fate Engine v2.1")
        self.geometry("1100x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="Scenario Director", 
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        for name, data in SCENARIOS.items():
            ctk.CTkButton(self.sidebar, text=name, fg_color=data.color,
                          command=lambda m=name: self.set_mode(m)).pack(pady=8, padx=20, fill="x")
        
        self.status_label = ctk.CTkLabel(self.sidebar, text=f"MODE: {self.current_mode}", 
                                         font=("Consolas", 14), text_color=THEME_COLOR)
        self.status_label.pack(side="bottom", pady=20)

    def _setup_plot(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.line, = self.ax.plot(range(HISTORY_LEN), self.engine.history, lw=3, color=THEME_COLOR)
        
        self.ax.set_ylim(-0.1, 1.2)
        self.ax.set_title("PROTEOMIC FLUX", color='white', weight='bold')
        
        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.get_tk_widget().grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.canvas = canvas

    def set_mode(self, mode: str):
        self.current_mode = mode
        self.status_label.configure(text=f"MODE: {mode.upper()}")

    def run_engine_loop(self):
        # 1. Get math from engine
        state = SCENARIOS[self.current_mode]
        current_val = self.engine.compute_next_step(state)
        
        # 2. Update visuals
        self.line.set_ydata(self.engine.history)
        color = THEME_COLOR if current_val < 0.59 else ALERT_COLOR
        self.line.set_color(color)
        
        self.canvas.draw_idle()
        self.after(REFRESH_MS, self.run_engine_loop)

if __name__ == "__main__":
    physics_engine = BiophysicsEngine()
    app = CellFateApp(physics_engine)
    app.mainloop()
