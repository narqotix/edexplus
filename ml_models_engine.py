# ==============================================================================
# EDEXPLUS AI — PYTHON MACHINE LEARNING PIPELINE & REST API
# Institution: DKTE Society's Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI-DS / AI-ML)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Frontend Integration

def generate_synthetic_data(n_samples=250):
    np.random.seed(42)
    se1 = np.random.randint(5, 25, n_samples)
    se2 = np.random.randint(5, 25, n_samples)
    internal = np.random.randint(15, 50, n_samples)
    viva = np.random.randint(15, 50, n_samples)
    attendance = np.random.randint(60, 100, n_samples)
    cgpa = np.random.uniform(5.5, 9.8, n_samples)
    leetcode = np.random.randint(0, 150, n_samples)
    
    hackathon_choices = ['Winner', 'Runner-Up', 'Finalist', 'Participant', 'None']
    hackathons = np.random.choice(hackathon_choices, n_samples, p=[0.1, 0.15, 0.2, 0.3, 0.25])
    
    # Calculate synthetic SEE marks based on mid-term performance
    see = np.clip((se1 * 0.8 + se2 * 0.8 + internal * 0.4 + np.random.normal(0, 2, n_samples)), 10, 50)
    
    df = pd.DataFrame({
        'SE1': se1,
        'SE2': se2,
        'Internal': internal,
        'Viva': viva,
        'SEE': see,
        'Attendance': attendance,
        'CGPA': cgpa,
        'LeetCode': leetcode,
        'Hackathon': hackathons
    })
    
    # Target Y for Risk Classification (0: LOW_RISK, 1: MODERATE_RISK, 2: HIGH_RISK)
    pct = ((df['SE1'] + df['SE2'] + df['SEE'] + df['Internal'] + df['Viva']) / 200) * 100
    df['Risk_Target'] = np.where((pct < 45) | (df['Attendance'] < 75), 2,
                        np.where((pct < 65) | (df['Attendance'] < 85), 1, 0))
    
    # Target Y for Placement Eligibility (1: Tier-1 Eligible, 0: Service Drives Eligible)
    df['Placement_Target'] = np.where((df['CGPA'] >= 8.0) & ((df['LeetCode'] >= 40) | (df['Hackathon'].isin(['Winner', 'Runner-Up']))), 1, 0)
    
    return df

# Initialize Data & Globally Trained Models
df_train = generate_synthetic_data()

X_risk = df_train[['SE1', 'SE2', 'Internal', 'Viva', 'Attendance']]
y_risk = df_train['Risk_Target']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_risk, y_risk)

encoder = OrdinalEncoder(categories=[['None', 'Participant', 'Finalist', 'Runner-Up', 'Winner']])
df_train['Hackathon_Encoded'] = encoder.fit_transform(df_train[['Hackathon']])

X_place = df_train[['CGPA', 'LeetCode', 'Hackathon_Encoded']]
y_place = df_train['Placement_Target']
scaler_place = StandardScaler()
X_place_scaled = scaler_place.fit_transform(X_place)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_place_scaled, y_place)

X_cluster = df_train[['LeetCode', 'Hackathon_Encoded']]
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(X_cluster)

X_grade = df_train[['SE1', 'SE2', 'Internal', 'Viva']]
y_grade = df_train['SEE']
reg_model = LinearRegression()
reg_model.fit(X_grade, y_grade)


@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    """Predicts student academic risk level using Random Forest."""
    data = request.json
    features = np.array([[data['se1'], data['se2'], data['internal'], data['viva'], data['attendance']]])
    prediction = rf_model.predict(features)[0]
    probabilities = rf_model.predict_proba(features)[0].tolist()
    
    risk_labels = {0: 'LOW_RISK', 1: 'MODERATE_RISK', 2: 'HIGH_RISK'}
    return jsonify({
        'predicted_risk': risk_labels[prediction],
        'confidence_scores': probabilities
    })

@app.route('/api/predict_placement', methods=['POST'])
def predict_placement():
    """Predicts placement drive eligibility tier using Logistic Regression."""
    data = request.json
    hack_input = data.get('hackathon', 'None')
    
    # Keyword parsing for hackathon distinctions
    hack_mapped = 'None'
    if 'winner' in hack_input.lower() or 'sih' in hack_input.lower():
        hack_mapped = 'Winner'
    elif 'runner' in hack_input.lower() or 'top 3' in hack_input.lower():
        hack_mapped = 'Runner-Up'
    elif 'finalist' in hack_input.lower():
        hack_mapped = 'Finalist'
    elif len(hack_input.strip()) > 0:
        hack_mapped = 'Participant'

    hack_encoded = encoder.transform([[hack_mapped]])[0][0]
    features = np.array([[data['cgpa'], data['leetcode'], hack_encoded]])
    features_scaled = scaler_place.transform(features)
    
    prob_tier1 = lr_model.predict_proba(features_scaled)[0][1]
    tier = 'Tier-1 / Dream Companies (10+ LPA)' if prob_tier1 > 0.5 else 'Core / Service Companies'
    
    return jsonify({
        'eligibility_tier': tier,
        'probability_score': round(prob_tier1, 4)
    })

@app.route('/api/forecast_grade', methods=['POST'])
def forecast_grade():
    """Forecasts final SEE exam mark and percentage using Linear Regression."""
    data = request.json
    features = np.array([[data['se1'], data['se2'], data['internal'], data['viva']]])
    predicted_see = reg_model.predict(features)[0]
    
    total_marks = data['se1'] + data['se2'] + predicted_see + data['internal'] + data['viva']
    predicted_pct = (total_marks / 200) * 100
    
    return jsonify({
        'forecasted_see_score': round(predicted_see, 2),
        'forecasted_total_percentage': round(predicted_pct, 2)
    })

if __name__ == '__main__':
    print("="*70)
    print("🚀 EDEXPLUS AI PYTHON ML ENGINE STARTED")
    print("   Running on http://127.0.0.1:5000")
    print("="*70)
    app.run(port=5000, debug=True)
