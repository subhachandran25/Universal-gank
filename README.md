# 🏦 Universal Bank — Personal Loan Analytics Dashboard

A full-stack Streamlit analytics dashboard to identify which customers are most likely to accept a personal loan offer.

## 🚀 Live Demo
Deploy instantly on [Streamlit Cloud](https://streamlit.io/cloud).

## 📊 Dashboard Tabs

| Tab | Type | Description |
|-----|------|-------------|
| 📊 Descriptive | Descriptive | Customer demographics, distributions, acceptance overview |
| 🔍 Diagnostic | Diagnostic | Key drivers, comparisons, correlation heatmap, service analysis |
| 🤖 Predictive | Predictive | Random Forest model, ROC curve, confusion matrix, feature importance |
| 💡 Prescriptive | Prescriptive | Target segments, sunburst drill-down, campaign recommendations |
| 🧮 Calculator | Interactive | Single customer loan propensity prediction tool |

## 🗂️ Project Structure

```
universal_bank_dashboard/
├── app.py               # Main Streamlit dashboard
├── charts.py            # All Plotly chart builder functions
├── data_loader.py       # Data loading & preprocessing
├── model.py             # Random Forest model training & prediction
├── UniversalBank.csv    # Dataset (add this to your repo)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## ⚙️ Run Locally

```bash
# Clone or download the repository
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `UniversalBank.csv` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo → Set `app.py` as the main file
4. Click **Deploy** ✅

## 📦 Dataset

The `UniversalBank.csv` file should be in the same directory as `app.py`.

| Column | Description |
|--------|-------------|
| Age | Customer age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| Family | Family size (1–4) |
| CCAvg | Avg monthly credit card spend ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced/Prof |
| Mortgage | Home mortgage value ($000) |
| Personal Loan | **Target variable** — did customer accept? (0/1) |
| Securities Account | Has securities account? (0/1) |
| CD Account | Has CD account? (0/1) |
| Online | Uses online banking? (0/1) |
| CreditCard | Has UniversalBank credit card? (0/1) |

## 🔑 Key Findings

- **Income** is the #1 predictor (correlation 0.50, feature importance 48.5%)
- **CD Account holders** accept loans at 46.4% vs 7.2% without one
- **Income > $100k** is the critical threshold for targeting
- **Graduate/Professional** education holders accept 3× more than undergrads
- Random Forest achieves **AUC = 0.9987** on the hold-out test set

## 🛠️ Tech Stack

- **Streamlit** — Dashboard framework
- **Plotly** — Interactive visualisations
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — Machine learning (Random Forest)
