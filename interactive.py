# interactive.py - Safe Routing GUI
# Run after main.py: python interactive.py

import os, sys, tkinter as tk
from tkinter import ttk, messagebox
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from src.routing import RiskGrid, compare_routes
from config import SHORELINE_LONS, SHORELINE_LATS

LOCS = {
    "Union Station": (43.6453, -79.3806),
    "CN Tower": (43.6426, -79.3871),
    "Yonge-Dundas Square": (43.6561, -79.3802),
    "Scarborough Town Centre": (43.7731, -79.2577),
    "Yorkdale Mall": (43.7254, -79.4515),
    "North York Centre": (43.7676, -79.4130),
    "Pearson Airport Area": (43.6800, -79.6100),
    "Danforth & Woodbine": (43.6920, -79.3270),
    "Etobicoke (Kipling)": (43.6440, -79.5100),
    "Beaches": (43.6670, -79.2930),
    "High Park": (43.6465, -79.4637),
    "Distillery District": (43.6503, -79.3596),
    "Liberty Village": (43.6382, -79.4209),
    "Bloor-Yonge": (43.6709, -79.3857),
    "St. Clair & Dufferin": (43.6810, -79.4350),
}
NAMES = list(LOCS.keys())
BG, PANEL = "#1a1a2e", "#16213e"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Toronto Safe Routing")
        root.configure(bg=BG)
        root.geometry("1100x700")

        if not os.path.exists("outputs/risk_grid.csv"):
            messagebox.showerror("Error", "Run main.py first.")
            sys.exit(1)

        self.grid = RiskGrid("outputs/risk_grid.csv")
        self.gdf = pd.read_csv("outputs/risk_grid.csv")

        s = ttk.Style()
        s.theme_use("clam")
        for name, bg, fg, font in [
            ("H.TLabel", PANEL, "#e0e0e0", ("Segoe UI", 11, "bold")),
            ("I.TLabel", PANEL, "#e0e0e0", ("Segoe UI", 10)),
            ("G.TLabel", PANEL, "#2ecc71", ("Segoe UI", 12, "bold")),
            ("R.TLabel", PANEL, "#e74c3c", ("Segoe UI", 12, "bold")),
            ("O.TLabel", PANEL, "#f39c12", ("Segoe UI", 12, "bold")),
            ("B.TLabel", PANEL, "#e0e0e0", ("Segoe UI", 20, "bold")),
        ]:
            s.configure(name, background=bg, foreground=fg, font=font)
        s.configure("P.TFrame", background=PANEL)
        s.configure("BG.TFrame", background=BG)
        s.configure("T.TLabel", background=BG, foreground="#e94560",
                     font=("Segoe UI", 16, "bold"))

        ttk.Label(root, text="Toronto Safe Routing", style="T.TLabel").pack(
            anchor="w", padx=20, pady=(10, 5))

        content = ttk.Frame(root, style="BG.TFrame")
        content.pack(fill="both", expand=True, padx=20, pady=5)

        left = ttk.Frame(content, style="BG.TFrame", width=340)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        ctrl = ttk.Frame(left, style="P.TFrame", padding=12)
        ctrl.pack(fill="x", pady=(0, 8))
        ttk.Label(ctrl, text="Start:", style="I.TLabel").pack(anchor="w", pady=(5, 2))
        self.sv = tk.StringVar(value="")
        cb_start = ttk.Combobox(ctrl, textvariable=self.sv, values=NAMES, width=32)
        cb_start.pack(fill="x")
        cb_start.set("Type an address or select...")
        
        ttk.Label(ctrl, text="End:", style="I.TLabel").pack(anchor="w", pady=(8, 2))
        
        self.ev = tk.StringVar(value="")
        cb_end = ttk.Combobox(ctrl, textvariable=self.ev, values=NAMES, width=32)
        cb_end.pack(fill="x")
        cb_end.set("Type an address or select...")
        ttk.Button(ctrl, text="FIND ROUTES", command=self._run).pack(fill="x", pady=(12, 5))
        self.status = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.status, style="I.TLabel",
                  wraplength=300).pack(anchor="w")

        self.res_frame = ttk.Frame(left, style="P.TFrame", padding=12)
        self.res_frame.pack(fill="both", expand=True)
        self.res_inner = ttk.Frame(self.res_frame, style="P.TFrame")
        self.res_inner.pack(fill="both", expand=True)

        mf = ttk.Frame(content, style="P.TFrame", padding=5)
        mf.pack(side="left", fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.fig.patch.set_facecolor(PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_base()

    def _draw_base(self):
        import contextily as cx
        self.ax.clear()
        self.ax.set_facecolor(BG)
        self.ax.scatter(self.gdf["glon"], self.gdf["glat"], c=self.gdf["route_risk"],
                        cmap="YlOrRd", alpha=0.4, s=12, vmin=0, vmax=1, zorder=2)
        self.ax.set_xlim(-79.65, -79.10)
        self.ax.set_ylim(43.58, 43.86)
        try:
            cx.add_basemap(self.ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.8, zorder=1, zoom=13)
        except Exception:
            pass # fallback if offline
        self.ax.set_xlabel("Longitude", color="#8899aa", fontsize=9)
        self.ax.set_ylabel("Latitude", color="#8899aa", fontsize=9)
        self.ax.set_title("Collision Risk Map", color="#e0e0e0", fontsize=12, fontweight="bold")
        self.ax.tick_params(colors="#8899aa", labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_color("#333355")
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_route(self, r, sn, en):
        self._draw_base()
        for path, color, ls, lw, label in [
            (r["shortest"]["path"], "#e74c3c", "--", 2.5,
             f'Shortest: {r["shortest"]["distance_km"]:.1f}km'),
            (r["safest"]["path"], "#2ecc71", "-", 3,
             f'Safest: {r["safest"]["distance_km"]:.1f}km'),
        ]:
            lats, lons = zip(*path)
            self.ax.plot(lons, lats, color=color, ls=ls, lw=lw,
                         alpha=0.9, label=label, zorder=3)

        sp, ep = r["shortest"]["path"][0], r["shortest"]["path"][-1]
        self.ax.scatter(sp[1], sp[0], c="#3498db", s=120, zorder=5,
                        edgecolors="white", lw=2)
        self.ax.scatter(ep[1], ep[0], c="#e67e22", s=150, zorder=5,
                        edgecolors="white", lw=2, marker="*")
        self.ax.annotate(sn, (sp[1], sp[0]), fontsize=7, fontweight="bold",
                         color="#3498db", xytext=(8, 10), textcoords="offset points")
        self.ax.annotate(en, (ep[1], ep[0]), fontsize=7, fontweight="bold",
                         color="#e67e22", xytext=(8, 10), textcoords="offset points")

        all_lats = [p[0] for p in r["shortest"]["path"] + r["safest"]["path"]]
        all_lons = [p[1] for p in r["shortest"]["path"] + r["safest"]["path"]]
        pad = 0.02
        self.ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
        self.ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)
        self.ax.set_title(f"{sn}  →  {en}", color="#e0e0e0", fontsize=12, fontweight="bold")
        self.ax.legend(loc="lower right", fontsize=9, facecolor=PANEL,
                       edgecolor="#333355", labelcolor="#e0e0e0")
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_results(self, r):
        for w in self.res_inner.winfo_children():
            w.destroy()
        sh, sa = r["shortest"], r["safest"]
        rr, ed = r["risk_reduction"] * 100, r["distance_increase"] * 100

        ttk.Label(self.res_inner, text="SHORTEST PATH", style="R.TLabel").pack(anchor="w", pady=(5, 2))
        ttk.Label(self.res_inner,
                  text=f"  {sh['distance_km']:.2f} km  |  Risk: {sh['risk_sum']:.3f}",
                  style="I.TLabel").pack(anchor="w")
        ttk.Separator(self.res_inner).pack(fill="x", pady=8)
        ttk.Label(self.res_inner, text="SAFEST PATH", style="G.TLabel").pack(anchor="w", pady=(0, 2))
        ttk.Label(self.res_inner,
                  text=f"  {sa['distance_km']:.2f} km  |  Risk: {sa['risk_sum']:.3f}",
                  style="I.TLabel").pack(anchor="w")
        ttk.Separator(self.res_inner).pack(fill="x", pady=8)
        ttk.Label(self.res_inner, text=f"{rr:.1f}%", style="B.TLabel").pack(anchor="w")
        ttk.Label(self.res_inner, text="risk reduction", style="G.TLabel").pack(anchor="w")
        ttk.Label(self.res_inner, text=f"+{ed:.1f}% extra distance",
                  style="O.TLabel").pack(anchor="w", pady=(8, 0))

    def _run(self):
        sn, en = self.sv.get().strip(), self.ev.get().strip()
        if not sn or not en or sn == "Type an address or select..." or en == "Type an address or select...":
            messagebox.showwarning("Error", "Please enter start and end locations.")
            return
        if sn == en:
            messagebox.showwarning("Error", "Start and end cannot be the same.")
            return

        self.status.set("Geocoding addresses...")
        self.root.update()

        def get_coords(loc_name):
            if loc_name in LOCS:
                return LOCS[loc_name]
            import osmnx as ox
            query = loc_name
            if "toronto" not in query.lower() and "mississauga" not in query.lower():
                query += ", Toronto, Ontario, Canada"
            try:
                # Returns (lat, lon)
                return ox.geocode(query)
            except Exception:
                return None

        start_loc = get_coords(sn)
        if not start_loc:
            self.status.set("Error: Could not find start address.")
            return

        end_loc = get_coords(en)
        if not end_loc:
            self.status.set("Error: Could not find end address.")
            return

        self.status.set("Computing routes...")
        self.root.update()
        try:
            r = compare_routes(self.grid, start_loc, end_loc)
            if not r:
                self.status.set("No path found.")
                return
            self._show_results(r)
            self._draw_route(r, sn, en)
            self.status.set(f"Risk reduced by {r['risk_reduction']*100:.1f}%")
        except Exception as e:
            self.status.set(f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()