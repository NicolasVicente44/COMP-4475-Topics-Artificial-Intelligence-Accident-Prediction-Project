import customtkinter as ctk

root = ctk.CTk()
sv = ctk.StringVar(value="")

def on_type(event):
    if event and event.keysym in ("Return", "Up", "Down", "Left", "Right"):
        return
    filtered = [x for x in ["Apple", "Banana", "Cherry"] if sv.get().lower() in x.lower()]
    cb.configure(values=filtered)
    
    if filtered:
        try:
            x = cb.winfo_rootx()
            y = cb.winfo_rooty() + cb.winfo_height()
            cb._dropdown_menu.open(x, y)
        except Exception as e:
            print("Error opening:", e)
    else:
        # try closing
        try:
            cb._dropdown_menu._withdraw()
        except:
            pass

cb = ctk.CTkComboBox(root, variable=sv, values=["Apple", "Banana", "Cherry"])
cb.pack()
cb.bind("<KeyRelease>", on_type)

root.after(2000, lambda: [sv.set("a"), on_type(None)])
root.after(3000, lambda: [sv.set("ap"), on_type(None)])
root.after(4000, root.destroy)
root.mainloop()
