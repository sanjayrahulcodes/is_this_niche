# 🎭 Is This Niche?

> "Tame Impala is so underground." — you, probably.

Find out how basic your music taste really is. Enter any song and get a brutal honest mainstream score from 0–100, plus a roast you probably deserve.

---

## What It Does

Most people think their music taste is more niche than it actually is.

**Is This Niche?** uses a Ridge Regression model trained on 100,000+ real Spotify tracks to score any song's mainstream-ness — then judges you accordingly.

| Score | Verdict |
|-------|---------|
| 85–100 | 💀 Ultra Mainstream |
| 65–84 | 🔥 Mainstream |
| 45–64 | 😐 Middle Ground |
| 25–44 | ✅ Actually Niche |
| 0–24 | 🧪 Deeply Underground |

---

## How To Run It

### 1. Clone the repo
git clone https://github.com/sanjayrahulcodes/is-this-niche.git
cd is-this-niche

### 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Get the dataset
Download from [Kaggle](https://kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
and place it in the `data/` folder as `tracks_features.csv`

### 4. Train the model
python train_model.py

### 5. Run the app
python app.py

---

## How It Works

1. A Ridge Regression model is trained on Spotify audio features — energy, danceability, acousticness, valence, tempo and more
2. For any song you search, it pulls the real Spotify popularity score from the dataset
3. It compares predicted vs actual popularity to determine the verdict
4. You get roasted accordingly

---

## Tech Stack
- **Python** — core language
- **scikit-learn** — Ridge Regression model
- **pandas / numpy** — data processing
- **tkinter** — desktop GUI
- **Dataset** — 114,000 Spotify tracks via Kaggle

---

## The Roast Is About You

The score isn't judging the song. It's judging your taste.

---

*Built as a supervised ML project using linear regression.*
