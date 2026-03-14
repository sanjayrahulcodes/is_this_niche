import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

print("📦 Loading dataset...")
df = pd.read_csv('data/tracks_features.csv')
print(f"Raw shape: {df.shape}")
print(df.columns.tolist())
# Features we use to predict popularity
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]
target = 'popularity'

# Clean data
df = df.dropna(subset=features + [target])
df = df[df['popularity'] > 0]
print(f"Clean shape: {df.shape}")

X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
print("\n🤖 Training model...")
model = Ridge(alpha=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Results:")
print(f"MAE:  ±{mae:.1f} popularity points")
print(f"R²:   {r2:.3f}")

# Feature importance plot
coef_df = pd.DataFrame({
    'Feature': features,
    'Impact': model.coef_
}).sort_values('Impact')


# Save model and scaler
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Also save the song database for live lookup
print("\n💾 Saving song database for lookup...")
df['name'] = df['track_name'].str.lower().str.strip()
df[['name', 'artists', 'popularity'] + features].to_csv('data/song_db.csv', index=False)
print("✅ Done! Run app.py to start the is this niche meter.")