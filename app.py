import tkinter as tk
from tkinter import font
import pickle
import numpy as np
import pandas as pd
import random
import threading

# ── Load model & data ──────────────────────────────────────────────────────────
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

song_db = pd.read_csv('data/song_db.csv')
song_db['name_clean']   = song_db['name'].str.lower().str.strip()
song_db['artist_clean'] = song_db['artists'].str.lower().str.strip()

FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

# ── Roast engine ───────────────────────────────────────────────────────────────
def get_verdict(score):
    if score >= 85:
        tier  = "ULTRA MAINSTREAM"
        emoji = "💀"
        color = "#ff4757"
        roasts = [
            "You found this on a Netflix trailer, didn't you.",
            "Your Spotify Wrapped is literally just the Billboard Hot 100.",
            "Congratulations. You ARE the algorithm.",
            "This song was in an iPhone commercial. Twice.",
        ]
    elif score >= 65:
        tier  = "MAINSTREAM"
        emoji = "🔥"
        color = "#ff6b35"
        roasts = [
            "You call this alternative. It has 800M streams.",
            "Your 'indie' playlist has a Spotify editorial cover.",
            "This is what people mean by 'I like all kinds of music'.",
            "You discovered this the same week everyone else did.",
        ]
    elif score >= 45:
        tier  = "MIDDLE GROUND"
        emoji = "😐"
        color = "#ffa502"
        roasts = [
            "Not boring, not interesting. The Switzerland of music taste.",
            "You're one festival away from going fully mainstream.",
            "Safely adventurous. Adventurously safe.",
            "Your music taste is a personality trait. Just not a strong one.",
        ]
    elif score >= 25:
        tier  = "ACTUALLY NICHE"
        emoji = "✅"
        color = "#2ed573"
        roasts = [
            "Okay. We'll allow it.",
            "You can mention this at parties. Once.",
            "Your Spotify Wrapped must genuinely confuse people.",
            "Not everyone gets it. That's the point, isn't it.",
        ]
    else:
        tier  = "DEEPLY UNDERGROUND"
        emoji = "🧪"
        color = "#1e90ff"
        roasts = [
            "Are you okay? Do you need someone to talk to?",
            "This artist's mom doesn't know about this release.",
            "You found this in a SoundCloud comment from 2013.",
            "Genuine, confused respect.",
        ]
    return tier, emoji, color, random.choice(roasts)

# ── Search logic ───────────────────────────────────────────────────────────────
def search_song(song_query, artist_query=""):
    song_q   = song_query.lower().strip()
    artist_q = artist_query.lower().strip()

    # Try exact match first, then partial
    exact   = song_db[song_db['name_clean'] == song_q]
    partial = song_db[song_db['name_clean'].str.contains(song_q, na=False, regex=False)]
    matches = exact if not exact.empty else partial

    if matches.empty:
        return None, "song_not_found"

    # Filter by artist if provided
    if artist_q:
        artist_matches = matches[matches['artist_clean'].str.contains(artist_q, na=False, regex=False)]
        if artist_matches.empty:
            song   = matches.sort_values('popularity', ascending=False).iloc[0]
            result = _build_result(song)
            result['warning'] = f"Artist '{artist_query}' not found — showing closest song match."
            return result, "artist_mismatch"
        matches = artist_matches

    song   = matches.sort_values('popularity', ascending=False).iloc[0]
    result = _build_result(song)
    return result, "ok"

def _build_result(song):
    actual    = song['popularity']
    fv        = np.array([[song[f] for f in FEATURES]])
    predicted = float(np.clip(model.predict(scaler.transform(fv))[0], 0, 100))
    tier, emoji, color, roast = get_verdict(actual)
    return {
        'name':      song['name'].title(),
        'artists':   song['artists'],
        'actual':    int(actual),
        'predicted': int(predicted),
        'tier':      tier,
        'emoji':     emoji,
        'color':     color,
        'roast':     roast,
        'warning':   None,
    }

# ── GUI ────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Is this niche?")
        self.geometry("560x740")
        self.resizable(False, False)
        self.configure(bg="#0a0a0f")

        self.fn_title = font.Font(family="Courier New", size=22, weight="bold")
        self.fn_sub   = font.Font(family="Courier New", size=10)
        self.fn_label = font.Font(family="Courier New", size=11, weight="bold")
        self.fn_body  = font.Font(family="Courier New", size=10)
        self.fn_score = font.Font(family="Courier New", size=48, weight="bold")
        self.fn_tier  = font.Font(family="Courier New", size=14, weight="bold")
        self.fn_roast = font.Font(family="Courier New", size=11, slant="italic")
        self.fn_warn  = font.Font(family="Courier New", size=9)

        self._build_ui()

    def _build_ui(self):
        # Header
        tk.Label(self, text="Is this niche?",
                 font=self.fn_title, bg="#0a0a0f", fg="#ffffff").pack(pady=(32, 2))
        tk.Label(self, text="find out how basic your taste really is",
                 font=self.fn_sub, bg="#0a0a0f", fg="#555566").pack(pady=(0, 24))

        wrap = tk.Frame(self, bg="#0a0a0f")
        wrap.pack(padx=40, fill="x")

        # Song name field
        tk.Label(wrap, text="SONG NAME", font=self.fn_label,
                 bg="#0a0a0f", fg="#555566").pack(anchor="w")
        song_frame = tk.Frame(wrap, bg="#1a1a2e",
                              highlightbackground="#333355", highlightthickness=1)
        song_frame.pack(fill="x", pady=(4, 12))
        self.song_entry = tk.Entry(song_frame, font=self.fn_label,
                                   bg="#1a1a2e", fg="#ffffff",
                                   insertbackground="#ffffff",
                                   relief="flat", bd=10)
        self.song_entry.pack(fill="x", ipady=8)
        self.song_entry.bind("<Return>", lambda e: self._on_search())
        self.song_entry.focus()

        # Artist field (optional)
        tk.Label(wrap, text="ARTIST  (optional)",
                 font=self.fn_label, bg="#0a0a0f", fg="#555566").pack(anchor="w")
        artist_frame = tk.Frame(wrap, bg="#1a1a2e",
                                highlightbackground="#333355", highlightthickness=1)
        artist_frame.pack(fill="x", pady=(4, 0))
        self.artist_entry = tk.Entry(artist_frame, font=self.fn_label,
                                     bg="#1a1a2e", fg="#ffffff",
                                     insertbackground="#ffffff",
                                     relief="flat", bd=10)
        self.artist_entry.pack(fill="x", ipady=8)
        self.artist_entry.bind("<Return>", lambda e: self._on_search())

        # Button
        self.btn = tk.Button(self, text="ANALYSE  →",
                             font=self.fn_label,
                             bg="#ffffff", fg="#0a0a0f",
                             activebackground="#cccccc",
                             relief="flat", cursor="hand2",
                             command=self._on_search)
        self.btn.pack(padx=40, fill="x", pady=(14, 0), ipady=10)

        # Divider
        tk.Frame(self, bg="#1a1a2e", height=1).pack(fill="x", padx=40, pady=24)

        # Result area
        self.result_frame = tk.Frame(self, bg="#0a0a0f")
        self.result_frame.pack(padx=40, fill="both", expand=True)
        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.result_frame.winfo_children():
            w.destroy()
        tk.Label(self.result_frame,
                 text="enter a song above\nto get your verdict",
                 font=self.fn_body, bg="#0a0a0f", fg="#333344",
                 justify="center").pack(expand=True)

    def _on_search(self):
        song   = self.song_entry.get().strip()
        artist = self.artist_entry.get().strip()
        if not song:
            return
        self.btn.config(state="disabled", text="searching...")
        threading.Thread(target=self._do_search,
                         args=(song, artist), daemon=True).start()

    def _do_search(self, song, artist):
        result, status = search_song(song, artist)
        self.after(0, lambda: self._show_result(result, status))

    def _show_result(self, result, status):
        self.btn.config(state="normal", text="ANALYSE  →")
        for w in self.result_frame.winfo_children():
            w.destroy()

        if status == "song_not_found":
            tk.Label(self.result_frame,
                     text="song not found.\ntry a different spelling.",
                     font=self.fn_body, bg="#0a0a0f", fg="#ff4757",
                     justify="center").pack(expand=True)
            return

        # Warning if artist mismatch
        if result.get('warning'):
            tk.Label(self.result_frame,
                     text=f"⚠  {result['warning']}",
                     font=self.fn_warn, bg="#0a0a0f", fg="#ffa502",
                     wraplength=460, justify="center").pack(pady=(0, 8))

        # Song + artist
        tk.Label(self.result_frame,
                 text=result['name'],
                 font=self.fn_label, bg="#0a0a0f", fg="#ffffff",
                 wraplength=460, justify="center").pack()
        tk.Label(self.result_frame,
                 text=result['artists'],
                 font=self.fn_body, bg="#0a0a0f", fg="#555566").pack(pady=(2, 14))

        # Big score
        tk.Label(self.result_frame,
                 text=str(result['actual']),
                 font=self.fn_score, bg="#0a0a0f",
                 fg=result['color']).pack()
        tk.Label(self.result_frame,
                 text="/ 100  MAINSTREAM SCORE",
                 font=self.fn_sub, bg="#0a0a0f", fg="#555566").pack(pady=(0, 14))

        # Score bar
        bar_bg = tk.Frame(self.result_frame, bg="#1a1a2e", height=8)
        bar_bg.pack(fill="x", pady=(0, 14))
        bar_bg.update_idletasks()
        bar_w = max(1, int(bar_bg.winfo_width() * result['actual'] / 100))
        tk.Frame(bar_bg, bg=result['color'], height=8, width=bar_w).place(x=0, y=0)

        # Tier badge
        badge = tk.Frame(self.result_frame, bg=result['color'])
        badge.pack(pady=(0, 10))
        tk.Label(badge,
                 text=f"  {result['emoji']}  {result['tier']}  ",
                 font=self.fn_tier, bg=result['color'], fg="#0a0a0f").pack(ipady=6)

        # Roast
        tk.Label(self.result_frame,
                 text=f'"{result["roast"]}"',
                 font=self.fn_roast, bg="#0a0a0f", fg="#aaaabb",
                 wraplength=460, justify="center").pack(pady=(4, 0))

        tk.Frame(self.result_frame, bg="#0a0a0f", height=20).pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()