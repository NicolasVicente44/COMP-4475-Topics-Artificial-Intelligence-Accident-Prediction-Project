# interactive.py - Safe Routing GUI
# Run after main.py: python interactive.py

import os, sys
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

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

# CTK theme colors for map embedded integration
BG = "#2b2b2b"

class App:
    def __init__(self, root):
        self.root = root
        root.title("Toronto AI Safe Routing")
        root.geometry("1300x800")
        
        # Grid layout configuration
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        if not os.path.exists("outputs/risk_grid.csv"):
            messagebox.showerror("Error", "Run main.py first to generate risk data.")
            sys.exit(1)

        self.grid = RiskGrid("outputs/risk_grid.csv")
        self.gdf = pd.read_csv("outputs/risk_grid.csv")

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self.root, width=360, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(8, weight=1)

        # Title/Logo
        self.logo = ctk.CTkLabel(self.sidebar, text="Navigation AI", font=ctk.CTkFont(size=28, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(30, 5))
        
        self.desc = ctk.CTkLabel(self.sidebar, text="Find the fastest vs safest routes.", font=ctk.CTkFont(size=14), text_color="gray")
        self.desc.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Start Entry
        self.sv = tk.StringVar(value="")
        self.lbl_start = ctk.CTkLabel(self.sidebar, text="Starting Location:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.lbl_start.grid(row=2, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.cb_start = ctk.CTkComboBox(self.sidebar, variable=self.sv, values=NAMES, width=320, height=35)
        self.cb_start.grid(row=3, column=0, padx=20, pady=(0, 10))
        self.cb_start.set("Type an address or select...")
        
        def clear_start(event):
            if self.sv.get() == "Type an address or select...":
                self.cb_start.set("")
        self.cb_start.bind("<Button-1>", clear_start)
        self.cb_start.bind("<FocusIn>", clear_start)

        # End Entry
        self.ev = tk.StringVar(value="")
        self.lbl_end = ctk.CTkLabel(self.sidebar, text="Destination:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.lbl_end.grid(row=4, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.cb_end = ctk.CTkComboBox(self.sidebar, variable=self.ev, values=NAMES, width=320, height=35)
        self.cb_end.grid(row=5, column=0, padx=20, pady=(0, 20))
        self.cb_end.set("Type an address or select...")

        def clear_end(event):
            if self.ev.get() == "Type an address or select...":
                self.cb_end.set("")
        self.cb_end.bind("<Button-1>", clear_end)
        self.cb_end.bind("<FocusIn>", clear_end)

        # Find Button
        self.btn_find = ctk.CTkButton(self.sidebar, text="FIND ROUTES", command=self._run, font=ctk.CTkFont(weight="bold"), height=45)
        self.btn_find.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="ew")

        # Status
        self.status = tk.StringVar(value="Ready.")
        self.lbl_status = ctk.CTkLabel(self.sidebar, textvariable=self.status, text_color="gray", wraplength=320)
        self.lbl_status.grid(row=7, column=0, padx=20, pady=5)

        # Dynamic Results Panel
        self.res_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.res_frame.grid(row=9, column=0, padx=20, pady=(10, 30), sticky="nsew")

        # --- Main Map Panel ---
        self.map_container = ctk.CTkFrame(self.root, corner_radius=15)
        self.map_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.patch.set_facecolor(BG) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
        
        self._draw_base()

    def _draw_base(self):
        import contextily as cx
        self.ax.clear()
        self.ax.set_facecolor(BG)
        # Plot Heatmap
        self.ax.scatter(self.gdf["glon"], self.gdf["glat"], c=self.gdf["route_risk"],
                        cmap="YlOrRd", alpha=0.3, s=12, vmin=0, vmax=1, zorder=2)
        self.ax.set_xlim(-79.65, -79.10)
        self.ax.set_ylim(43.58, 43.86)
        try:
            cx.add_basemap(self.ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.8, zorder=1, zoom=13)
        except Exception:
            pass
            
        self.ax.set_xlabel("Longitude", color="#8899aa", fontsize=10)
        self.ax.set_ylabel("Latitude", color="#8899aa", fontsize=10)
        self.ax.set_title("Collision Risk Intelligent Map", color="#e0e0e0", fontsize=14, fontweight="bold")
        self.ax.tick_params(colors="#8899aa", labelsize=9)
        for sp in self.ax.spines.values():
            sp.set_color("#444466")
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_route(self, r, sn, en):
        self._draw_base()
        
        for path, color, ls, lw, label in [
            (r["shortest"]["path"], "#e74c3c", "--", 2.5, f"Shortest: {r['shortest']['distance_km']:.1f}km"),
            (r["safest"]["path"], "#2ecc71", "-", 3.5, f"Safest: {r['safest']['distance_km']:.1f}km"),
        ]:
            lats, lons = zip(*path)
            self.ax.plot(lons, lats, color=color, ls=ls, lw=lw, alpha=0.9, label=label, zorder=3)

        sp, ep = r["shortest"]["path"][0], r["shortest"]["path"][-1]
        
        # Start marker
        self.ax.scatter(sp[1], sp[0], c="#3498db", s=140, zorder=5, edgecolors="white", lw=2)
        # End marker
        self.ax.scatter(ep[1], ep[0], c="#f39c12", s=180, zorder=5, edgecolors="white", lw=2, marker="*")
        
        format_sn = sn if len(sn) < 25 else sn[:25] + "..."
        format_en = en if len(en) < 25 else en[:25] + "..."
        
        self.ax.annotate(format_sn, (sp[1], sp[0]), fontsize=9, fontweight="bold",
                         color="#3498db", xytext=(10, 10), textcoords="offset points")
        self.ax.annotate(format_en, (ep[1], ep[0]), fontsize=9, fontweight="bold",
                         color="#f39c12", xytext=(10, 10), textcoords="offset points")

        all_lats = [p[0] for p in r["shortest"]["path"] + r["safest"]["path"]]
        all_lons = [p[1] for p in r["shortest"]["path"] + r["safest"]["path"]]
        pad = 0.02
        self.ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
        self.ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)
        self.ax.set_title(f"Route: {format_sn}  →  {format_en}", color="#e0e0e0", fontsize=14, fontweight="bold")
        
        self.ax.legend(loc="lower right", fontsize=10, facecolor=BG, edgecolor="#444466", labelcolor="#e0e0e0")
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_results(self, r):
        for w in self.res_frame.winfo_children():
            w.destroy()
            
        sh, sa = r["shortest"], r["safest"]
        rr, ed = r["risk_reduction"] * 100, r["distance_increase"] * 100

        # Create aesthetic cards
        ctk.CTkLabel(self.res_frame, text="SHORTEST PATH", font=ctk.CTkFont(size=14, weight="bold"), text_color="#e74c3c").pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self.res_frame, text=f"• Distance: {sh['distance_km']:.2f} km\n• Risk Score: {sh['risk_sum']:.3f}", 
                     justify="left", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=10)
        
        ctk.CTkFrame(self.res_frame, height=2, fg_color="#444466").pack(fill="x", pady=15)
        
        ctk.CTkLabel(self.res_frame, text="SAFEST PATH", font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ecc71").pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self.res_frame, text=f"• Distance: {sa['distance_km']:.2f} km\n• Risk Score: {sa['risk_sum']:.3f}", 
                     justify="left", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=10)
        
        ctk.CTkFrame(self.res_frame, height=2, fg_color="#444466").pack(fill="x", pady=15)
        
        # Trade-off Analysis Highlights
        ctk.CTkLabel(self.res_frame, text=f"{rr:.1f}%", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w", pady=(5,0))
        ctk.CTkLabel(self.res_frame, text="Overall Risk Reduction", font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ecc71").pack(anchor="w")
        
        ctk.CTkLabel(self.res_frame, text=f"(+{ed:.1f}% extra travel distance)", font=ctk.CTkFont(size=12), text_color="#f39c12").pack(anchor="w", pady=(5, 0))

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

        self.status.set("Computing A* mathematical routes...")
        self.root.update()
        try:
            r = compare_routes(self.grid, start_loc, end_loc)
            if not r:
                self.status.set("No path found.")
                return
            self._show_results(r)
            self._draw_route(r, sn, en)
            self.status.set(f"Done! Risk dynamically reduced by {r['risk_reduction']*100:.1f}%")
        except Exception as e:
            self.status.set(f"Error: {e}")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root)
    root.mainloop()