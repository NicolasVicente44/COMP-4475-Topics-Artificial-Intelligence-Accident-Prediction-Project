import os, sys
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from src.routing import RiskGrid, compare_routes

# Predefined locations for the dropdown menu
LOCS = {
    "Union Station": (43.6453, -79.3806), "CN Tower": (43.6426, -79.3871),
    "Yonge-Dundas Square": (43.6561, -79.3802), "Scarborough Town Centre": (43.7731, -79.2577),
    "Yorkdale Mall": (43.7254, -79.4515), "North York Centre": (43.7676, -79.4130),
    "Pearson Airport Area": (43.6800, -79.6100), "Danforth & Woodbine": (43.6920, -79.3270),
    "Etobicoke (Kipling)": (43.6440, -79.5100), "Beaches": (43.6670, -79.2930),
    "High Park": (43.6465, -79.4637), "Distillery District": (43.6503, -79.3596),
    "Liberty Village": (43.6382, -79.4209), "Bloor-Yonge": (43.6709, -79.3857),
    "St. Clair & Dufferin": (43.6810, -79.4350),
}
NAMES = list(LOCS.keys())
BG = "#2b2b2b"
PH = "Type an address or select..."

# The main application class for the interactive map and route comparison tool  
class App:
    # Initialize the application
    def __init__(self, root):
        self.root = root
        root.title("Toronto Safe Routing")
        root.geometry("1300x800")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        if not os.path.exists("outputs/risk_grid.csv"):
            messagebox.showerror("Error", "Run main.py first to generate risk data.")
            sys.exit(1)

        self.grid = RiskGrid("outputs/risk_grid.csv")
        self.gdf = pd.read_csv("outputs/risk_grid.csv")
        self._scroll_timer = None
        self._pan_start = None
        self._build_sidebar()
        self._build_map()
        self._draw_base()

    # Helper function to build the sidebar
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self.root, width=360, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(8, weight=1)
        bold = ctk.CTkFont(weight="bold")
        self.sv, self.ev = tk.StringVar(), tk.StringVar()
        for row, label, var in [(2, "Starting Location:", self.sv), (4, "Destination:", self.ev)]:
            ctk.CTkLabel(sb, text=label, anchor="w", font=bold).grid(row=row, column=0, padx=20, pady=(15, 5), sticky="w")
            cb = ctk.CTkComboBox(sb, variable=var, values=NAMES, width=320, height=35)
            cb.grid(row=row + 1, column=0, padx=20, pady=(0, 10))
            cb.set(PH)
            cb.bind("<Button-1>", lambda e, v=var, c=cb: c.set("") if v.get() == PH else None)
            cb.bind("<FocusIn>", lambda e, v=var, c=cb: c.set("") if v.get() == PH else None)
        ctk.CTkButton(sb, text="FIND ROUTES", command=self._run, font=bold, height=45).grid(row=6, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.status = tk.StringVar(value="Ready.")
        ctk.CTkLabel(sb, textvariable=self.status, text_color="gray", wraplength=320).grid(row=7, column=0, padx=20, pady=5)
        self.res_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self.res_frame.grid(row=9, column=0, padx=20, pady=(10, 30), sticky="nsew")

    # Helper function to build the map
    def _build_map(self):
        mc = ctk.CTkFrame(self.root, corner_radius=15)
        mc.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mc)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=(15, 0))
        tf = tk.Frame(mc, bg=BG)
        tf.pack(fill="x", padx=15, pady=(0, 10))
        self.toolbar = NavigationToolbar2Tk(self.canvas, tf)
        self.toolbar.config(background=BG)
        for child in self.toolbar.winfo_children():
            try: child.config(background=BG, foreground="#cccccc")
            except tk.TclError: pass
        self.toolbar.update()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # Helper function to debounce the draw event
    def _debounce_draw(self, ms=80):
        if self._scroll_timer: self.root.after_cancel(self._scroll_timer)
        self._scroll_timer = self.root.after(ms, self._do_draw)

    def _do_draw(self):
        self._scroll_timer = None
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        s = 0.8 if event.button == "up" else 1.25
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        rx, ry = (event.xdata - xl[0]) / (xl[1] - xl[0]), (event.ydata - yl[0]) / (yl[1] - yl[0])
        nw, nh = (xl[1] - xl[0]) * s, (yl[1] - yl[0]) * s
        self.ax.set_xlim(event.xdata - nw * rx, event.xdata + nw * (1 - rx))
        self.ax.set_ylim(event.ydata - nh * ry, event.ydata + nh * (1 - ry))
        self._debounce_draw()

    def _on_press(self, event):
        if event.inaxes == self.ax and event.button == 1 and not self.toolbar.mode:
            self._pan_start = (event.xdata, event.ydata)

    def _on_release(self, event):
        if self._pan_start:
            self._pan_start = None
            self.canvas.draw_idle()

    def _on_motion(self, event):
        if not self._pan_start or event.inaxes != self.ax: return
        dx, dy = self._pan_start[0] - event.xdata, self._pan_start[1] - event.ydata
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(xl[0] + dx, xl[1] + dx)
        self.ax.set_ylim(yl[0] + dy, yl[1] + dy)
        self._debounce_draw(50)

    def _style_ax(self, title="Collision Risk Intelligent Map"):
        self.ax.set_xlabel("Longitude", color="#8899aa", fontsize=10)
        self.ax.set_ylabel("Latitude", color="#8899aa", fontsize=10)
        self.ax.set_title(title, color="#e0e0e0", fontsize=14, fontweight="bold")
        self.ax.tick_params(colors="#8899aa", labelsize=9)
        for sp in self.ax.spines.values(): sp.set_color("#444466")
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_base(self):
        import contextily as cx
        self.ax.clear()
        self.ax.set_facecolor(BG)
        self.ax.scatter(self.gdf["glon"], self.gdf["glat"], c=self.gdf["route_risk"],
                        cmap="YlOrRd", alpha=0.3, s=12, vmin=0, vmax=1, zorder=2, rasterized=True)
        self.ax.set_xlim(-79.65, -79.10)
        self.ax.set_ylim(43.58, 43.86)
        try:
            cx.add_basemap(self.ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik,
                           alpha=0.8, zorder=1, zoom=13)
        except Exception: pass
        self._style_ax()

    # Helper function to draw the route and display the results 
    def _draw_route(self, r, sn, en):
        self._draw_base()
        for key, clr, ls, lw, fmt in [
            ("shortest", "#e74c3c", "--", 2.5, "distance_km"), ("safest", "#2ecc71", "-", 3.5, "distance_km"),
            ("fastest", "#f1c40f", "-.", 3.0, "time_mins")]:
            lats, lons = zip(*r[key]["path"])
            unit = "km" if fmt == "distance_km" else "m"
            self.ax.plot(lons, lats, color=clr, ls=ls, lw=lw, alpha=0.9, zorder=3,
                         label=f"{key.title()}: {r[key][fmt]:.1f}{unit}")
        sp, ep = r["shortest"]["path"][0], r["shortest"]["path"][-1]
        self.ax.scatter(sp[1], sp[0], c="#3498db", s=140, zorder=5, edgecolors="white", lw=2)
        self.ax.scatter(ep[1], ep[0], c="#f39c12", s=180, zorder=5, edgecolors="white", lw=2, marker="*")
        fsn, fen = (sn[:25] + "..." if len(sn) >= 25 else sn), (en[:25] + "..." if len(en) >= 25 else en)
        for pt, lbl, clr in [(sp, fsn, "#3498db"), (ep, fen, "#f39c12")]:
            self.ax.annotate(lbl, (pt[1], pt[0]), fontsize=9, fontweight="bold", color=clr,
                             xytext=(10, 10), textcoords="offset points")
        all_pts = r["shortest"]["path"] + r["safest"]["path"] + r["fastest"]["path"]
        lats, lons = zip(*all_pts)
        self.ax.set_xlim(min(lons) - 0.02, max(lons) + 0.02)
        self.ax.set_ylim(min(lats) - 0.02, max(lats) + 0.02)
        self.ax.legend(loc="lower right", fontsize=10, facecolor=BG, edgecolor="#444466", labelcolor="#e0e0e0")
        self._style_ax(f"Route: {fsn}  →  {fen}")

    # Helper function to add a card to the results frame
    def _add_card(self, title, color, data):
        ctk.CTkLabel(self.res_frame, text=title, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=color).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self.res_frame, justify="left", font=ctk.CTkFont(size=13),
                     text=f"• Time: {data['time_mins']:.1f}m  • Dist: {data['distance_km']:.2f}km  • Risk: {data['risk_sum']:.3f}"
                     ).pack(anchor="w", padx=10)
        ctk.CTkFrame(self.res_frame, height=2, fg_color="#444466").pack(fill="x", pady=10)

    # Helper function to show the results and display the risk reduction
    def _show_results(self, r):
        for w in self.res_frame.winfo_children(): w.destroy()
        self._add_card("FASTEST PATH", "#f1c40f", r["fastest"])
        self._add_card("SAFEST PATH", "#2ecc71", r["safest"])
        self._add_card("SHORTEST PATH", "#e74c3c", r["shortest"])
        rr = r["risk_reduction"] * 100
        ctk.CTkLabel(self.res_frame, text=f"{rr:.1f}%", font=ctk.CTkFont(size=32, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(self.res_frame, text="Overall Risk Reduction",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ecc71").pack(anchor="w")
        dt = r["safest"]["time_mins"] - r["fastest"]["time_mins"]
        ctk.CTkLabel(self.res_frame, text=f"(+{dt:.1f} mins vs fastest)",
                     font=ctk.CTkFont(size=12), text_color="#f39c12").pack(anchor="w", pady=(5, 0))

    # Helper function to get coordinates from an address using OSMnx 
    def _get_coords(self, name):
        if name in LOCS: return LOCS[name]
        import osmnx as ox
        q = name if ("toronto" in name.lower() or "mississauga" in name.lower()) else name + ", Toronto, Ontario, Canada"
        try: return ox.geocode(q)
        except Exception: return None

    # Helper function to run the route comparison and display the results 
    def _run(self):
        sn, en = self.sv.get().strip(), self.ev.get().strip()
        if not sn or not en or sn == PH or en == PH:
            messagebox.showwarning("Error", "Please enter start and end locations."); return
        if sn == en:
            messagebox.showwarning("Error", "Start and end cannot be the same."); return
        self.status.set("Geocoding addresses...")
        self.root.update()
        start_loc = self._get_coords(sn)
        if not start_loc: self.status.set("Error: Could not find start address."); return
        end_loc = self._get_coords(en)
        if not end_loc: self.status.set("Error: Could not find end address."); return
        self.status.set("Computing A* route...")
        self.root.update()
        try:
            r = compare_routes(self.grid, start_loc, end_loc)
            if not r: self.status.set("No path found."); return
            self._show_results(r)
            self._draw_route(r, sn, en)
            self.status.set(f"Risk reduced by {r['risk_reduction']*100:.1f}%")
        except Exception as e:
            self.status.set(f"Error: {e}")

 # Main entry point to run the application for the interactive GUI
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root)
    root.mainloop()
