# 🏏 IPL Score Analyzer — Streamlit App

Live web app to analyze IPL match data and predict scores & winners.

---

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1: Upload to GitHub

1. Go to [github.com](https://github.com) → sign in
2. Click **"New repository"**
3. Name it: `ipl-analyzer`
4. Click **"Create repository"**
5. Upload all files from this folder:
   - `app.py`
   - `requirements.txt`
   - `ipl_matches.csv`

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `ipl-analyzer`
5. Main file: `app.py`
6. Click **"Deploy!"**

Done! You'll get a live URL like:
`https://your-name-ipl-analyzer.streamlit.app`

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

---

## 📄 Pages in the App

| Page | What it shows |
|---|---|
| Dashboard | Wins, score distributions, seasonal trends, heatmap |
| Team Analysis | Deep dive per team — head-to-head, win rates |
| Score Predictor | Select teams & venue → predicts score + win % |
| ML Insights | Feature importance, ANOVA, correlation heatmap |

---

## 🔧 Libraries Used

`streamlit` `pandas` `numpy` `plotly` `scipy` `scikit-learn`
