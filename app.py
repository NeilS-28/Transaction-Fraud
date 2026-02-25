


import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import PyPDF2
import re
from sklearn.metrics import roc_curve, auc, recall_score
import time
from datetime import datetime

# --- Model Paths ---
DT_PATH = 'decision_tree_model.joblib'
RF_PATH = 'random_Forest_model.pkl'
XGB_PATH = 'XGBoost_model.joblib'
LR_PATH = 'logistic_regression_model.joblib'
SCALER_PKL = 'scaler.pkl'
SCALER_JOBLIB = 'scaler.joblib'
ALLOWED_EXTENSIONS = ['csv', 'pdf']

@st.cache_resource
def load_models():
    try:
        dt = joblib.load(DT_PATH)
        rf = joblib.load(RF_PATH)
        xgb = joblib.load(XGB_PATH)
        lr = joblib.load(LR_PATH)
        try:
            scaler = joblib.load(SCALER_PKL)
        except:
            scaler = joblib.load(SCALER_JOBLIB)
        return dt, rf, xgb, lr, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

dt_model, rf_model, xgb_model, lr_model, scaler = load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_csv_from_pdf(pdf_file):
    try:
        pdfreader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdfreader.pages:
            text += page.extract_text()
        lines = text.strip().split('\n')
        data = []
        for line in lines:
            row = re.split(r',\s*', line.strip())
            if len(row) > 1:
                data.append(row)
        if data:
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        else:
            return None
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

def preprocess_data(df):
    try:
        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if 'Time' in missing_cols:
                required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return None, None, f"Missing required columns: {', '.join(missing_cols)}"
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if len(df) == 0:
            return None, None, "No valid data found after cleaning."
        y_true = None
        if 'Class' in df.columns:
            df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
            y_true = df['Class'].values
        feature_cols = required_cols
        X = df[feature_cols].copy()
        X_scaled = scaler.transform(X)
        return X_scaled, y_true, None
    except Exception as e:
        return None, None, f"Preprocessing error: {str(e)}"

def predict_fraud(X_scaled, model):
    try:
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        return predictions.astype(int), probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def generate_demo_transaction():
    merchants = ["Amazon", "Walmart", "Target", "Starbucks", "McDonald's", "Shell Gas", "Apple Store", "Best Buy", "Netflix", "Uber"]
    t_id = f"TXN{np.random.randint(100000, 999999)}"
    amount = np.random.choice([np.random.uniform(5, 500), np.random.uniform(1000, 5000) if np.random.random() < 0.15 else np.random.uniform(5, 500)])
    is_fraud = np.random.random() < 0.20
    fraud_prob = np.random.uniform(0.7, 0.99) if is_fraud else np.random.uniform(0.01, 0.35)
    merchant = np.random.choice(merchants)
    card_last4 = f"****{np.random.randint(1000, 9999)}"
    status = 'pending'
    return dict(id=t_id, datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), amount=amount, merchant=merchant, card=card_last4,
                fraud_prob=fraud_prob, is_fraud=is_fraud, status=status, created_at=time.time())

if 'demo_transactions' not in st.session_state:
    st.session_state.demo_transactions = []
    st.session_state.last_update = time.time()
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.show_results = False

st.set_page_config(page_title="Fraud Detection", layout="wide")
main_col, dashboard_col = st.columns([2, 1])

# ========= LEFT: APP UI & RESULTS =========
with main_col:
    st.title("Fraud Detection System")
    st.markdown("### CREDIT CARD TRANSACTION ANALYSIS")
    st.info(
        "How to Use:\n- Upload your credit card transaction data (CSV or PDF)\n- File must have columns: Time, V1-V28, Amount\n"
        "- Optional: Include 'Class' column (0=Legitimate, 1=Fraud) for ROC AUC and Recall metrics\n"
        "- Click Check Transaction to analyze with all 4 models")

    uploaded_file = st.file_uploader("Choose your file (CSV or PDF)", type=ALLOWED_EXTENSIONS)
    check = st.button("Check Transaction", use_container_width=True)

    if check and uploaded_file is not None:
        filename = uploaded_file.name
        if not allowed_file(filename):
            st.error("Invalid file format. Please upload a CSV or PDF.")
        else:
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = extract_csv_from_pdf(uploaded_file)
                if df is None:
                    st.error("Could not extract data from the uploaded file.")
                else:
                    X_scaled, y_true, error = preprocess_data(df)
                    if error:
                        st.error(error)
                    else:
                        models = {
                            "Decision Tree": dt_model,
                            "Random Forest": rf_model,
                            "XGBoost": xgb_model,
                            "Logistic Regression": lr_model,
                        }
                        preds, probs = {}, {}
                        for name, model in models.items():
                            pred, prob = predict_fraud(X_scaled, model)
                            preds[name] = pred
                            probs[name] = prob
                        total_transactions = len(next(iter(preds.values())))
                        recall_scores, roc_aucs, fprs, tprs = {}, {}, {}, {}
                        if y_true is not None:
                            for name in preds:
                                recall_scores[name] = recall_score(y_true, preds[name])
                                fprs[name], tprs[name], _ = roc_curve(y_true, probs[name])
                                roc_aucs[name] = auc(fprs[name], tprs[name])
                        avg_probs = {k: float(np.mean(probs[k])) for k in probs}
                        st.session_state.analysis_results = {
                            "names": list(models.keys()),
                            "preds": {k: preds[k].tolist() for k in preds},
                            "probs": {k: probs[k].tolist() for k in probs},
                            "y_true": y_true.tolist() if y_true is not None else None,
                            "recalls": recall_scores,
                            "aucs": roc_aucs,
                            "fprs": {k: fprs[k].tolist() for k in fprs} if y_true is not None else {},
                            "tprs": {k: tprs[k].tolist() for k in tprs} if y_true is not None else {},
                            "totals": total_transactions,
                            "avg_probs": avg_probs,
                        }
                        st.session_state.show_results = True
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

    # Display results after analysis and on every rerun!
    if st.session_state.show_results and st.session_state.analysis_results:
        res = st.session_state.analysis_results
        models = res["names"]
        preds = {k: np.array(res['preds'][k]) for k in models}
        probs = {k: np.array(res['probs'][k]) for k in models}
        total_transactions = res['totals']
        y_true = np.array(res['y_true']) if res["y_true"] is not None else None
        recalls, aucs = res["recalls"], res["aucs"]
        colors = ['#6a82fb', '#ffb800', '#FC8181', '#636e72']

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        st.success("✅ Transaction analysis completed successfully!")
        
        st.subheader("📈 Overall Summary")
        cols = st.columns(len(models) + 1)
        cols[0].metric("Total Transactions", total_transactions)
        for i, name in enumerate(models):
            fraud_count = int(np.sum(preds[name]))
            fraud_pct = (fraud_count/total_transactions)*100
            cols[i+1].metric(f"{name} Fraud", fraud_count, f"{fraud_pct:.1f}%")

        st.subheader("🔍 Fraud Detection Distribution")
        fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
        for i, name in enumerate(models):
            fraud_count = np.sum(preds[name])
            legit_count = total_transactions - fraud_count
            axes1[i].pie([fraud_count, legit_count],
                labels=['Fraudulent', 'Legitimate'], autopct='%1.1f%%', colors=[colors[i], '#dfe6e9'],
                startangle=90, textprops={'color':'#232946', 'fontweight':'bold', 'fontsize': 10})
            axes1[i].set_title(f"{name}\n({int(fraud_count)} fraudulent)", fontweight='bold', fontsize=11)
        fig1.tight_layout()
        st.pyplot(fig1)

        # Probability distribution comparison
        st.subheader("📉 Fraud Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        for name, color in zip(models, colors):
            ax2.hist(probs[name], bins=40, alpha=0.6, label=name, edgecolor='black', color=color, linewidth=0.8)
        ax2.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax2.set_title('Fraud Probability Distribution Across Models', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim([0, 1])
        fig2.tight_layout()
        st.pyplot(fig2)

        # Performance metrics section (only if ground truth available)
        if y_true is not None:
            st.subheader("🎯 Model Performance Metrics")
            
            # Display metrics in columns
            metric_cols = st.columns(len(models))
            for i, name in enumerate(models):
                with metric_cols[i]:
                    st.markdown(f"**{name}**")
                    st.metric("Recall", f"{recalls[name]:.4f}")
                    st.metric("ROC AUC", f"{aucs[name]:.4f}")
                    st.metric("Avg Fraud Prob", f"{res['avg_probs'][name]:.4f}")
            
            # ROC Curve Comparison
            st.markdown("#### ROC Curve Analysis")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for name, color in zip(models, colors):
                ax3.plot(np.array(res["fprs"][name]), np.array(res["tprs"][name]), 
                        color=color, lw=2.5, label=f"{name} (AUC = {aucs[name]:.4f})")
            ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax3.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
            ax3.set_title('ROC Curve - Model Comparison', fontsize=14, fontweight='bold')
            ax3.legend(loc="lower right", fontsize=10)
            ax3.grid(True, alpha=0.3, linestyle='--')
            fig3.tight_layout()
            st.pyplot(fig3)

            # Recall Score Comparison
            st.markdown("#### Recall Score Comparison")
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            recall_vals = [recalls[n] for n in models]
            bars = ax4.bar(models, recall_vals, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            for bar, recall in zip(bars, recall_vals):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                        f'{recall:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1.15])
            ax4.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
            ax4.set_title('Recall Score - Model Comparison (Higher is Better)', fontsize=14, fontweight='bold')
            ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
            plt.xticks(rotation=15, ha='right')
            fig4.tight_layout()
            st.pyplot(fig4)
            
            # Combined metrics visualization
            st.markdown("#### Combined Performance Overview")
            fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Recall vs ROC AUC scatter
            recall_vals = [recalls[n] for n in models]
            auc_vals = [aucs[n] for n in models]
            for i, (name, color) in enumerate(zip(models, colors)):
                ax5a.scatter(recall_vals[i], auc_vals[i], s=300, color=color, 
                           edgecolor='black', linewidth=2, alpha=0.7, label=name)
                ax5a.annotate(name, (recall_vals[i], auc_vals[i]), 
                            fontsize=9, ha='center', va='center', fontweight='bold')
            ax5a.set_xlabel('Recall Score', fontsize=11, fontweight='bold')
            ax5a.set_ylabel('ROC AUC Score', fontsize=11, fontweight='bold')
            ax5a.set_title('Recall vs ROC AUC', fontsize=12, fontweight='bold')
            ax5a.grid(True, alpha=0.3, linestyle='--')
            ax5a.set_xlim([min(recall_vals)-0.05, 1.05])
            ax5a.set_ylim([min(auc_vals)-0.05, 1.05])
            
            # Detection rate comparison
            fraud_detected = [np.sum(preds[n]) for n in models]
            ax5b.barh(models, fraud_detected, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            for i, count in enumerate(fraud_detected):
                ax5b.text(count + max(fraud_detected)*0.02, i, str(int(count)), 
                         va='center', fontsize=11, fontweight='bold')
            ax5b.set_xlabel('Fraudulent Transactions Detected', fontsize=11, fontweight='bold')
            ax5b.set_title('Fraud Detection Count', fontsize=12, fontweight='bold')
            ax5b.grid(True, axis='x', alpha=0.3, linestyle='--')
            
            fig5.tight_layout()
            st.pyplot(fig5)
        else:
            st.info("ℹ️ Include 'Class' column in your data (0=Legitimate, 1=Fraud) to view Recall and ROC AUC metrics.")

        st.subheader("📋 Detailed Model Comparison Table")
        comp_data = {
            'Model': models,
            'Fraudulent': [int(np.sum(preds[n])) for n in models],
            'Legitimate': [int(total_transactions - np.sum(preds[n])) for n in models],
            'Fraud %': [f"{(np.sum(preds[n])/total_transactions)*100:.2f}%" for n in models],
            'Avg Fraud Prob': [f"{res['avg_probs'][n]:.4f}" for n in models],
        }
        if y_true is not None:
            comp_data['Recall Score'] = [f"{recalls[n]:.4f}" for n in models]
            comp_data['ROC AUC Score'] = [f"{aucs[n]:.4f}" for n in models]
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Add download button for results
        csv = comp_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        st.caption("💡 Results are cached and will persist even when the live dashboard refreshes.")

# ========= RIGHT: LIVE DASHBOARD =========
with dashboard_col:
    st_autorefresh(interval=3000, key="fraud_dashboard")
    st.markdown("### 🎯 Live Fraud Transaction Dashboard")
    st.caption("Demo of real-time transaction feed (refreshes independently)")

    current_time = time.time()
    if len(st.session_state.demo_transactions) == 0 or (current_time - st.session_state.last_update) > 3:
        st.session_state.demo_transactions.insert(0, generate_demo_transaction())
        if len(st.session_state.demo_transactions) > 30:
            st.session_state.demo_transactions = st.session_state.demo_transactions[:30]
        st.session_state.last_update = current_time

    for txn in st.session_state.demo_transactions:
        age = time.time() - txn['created_at']
        if txn['status'] == 'pending':
            if txn['fraud_prob'] > 0.85:
                txn['status'] = 'blocked'
            elif txn['fraud_prob'] > 0.65 or txn['is_fraud']:
                txn['status'] = 'reviewing'
            else:
                txn['status'] = 'approved'
        if txn['status'] == 'reviewing' and age > 3:
            txn['status'] = "blocked" if txn['is_fraud'] else "approved"

    n_total = len(st.session_state.demo_transactions)
    n_fraud = sum(tx['is_fraud'] for tx in st.session_state.demo_transactions)
    n_blocked = sum(tx['status'] == "blocked" for tx in st.session_state.demo_transactions)
    n_legit = sum(tx['status'] == "approved" for tx in st.session_state.demo_transactions)
    n_review = sum(tx['status'] == "reviewing" for tx in st.session_state.demo_transactions)

    st.markdown("#### 📊 Stats")
    dash_cols = st.columns(5)
    dash_cols[0].metric("Total", n_total)
    dash_cols[1].metric("Fraud", n_fraud)
    dash_cols[2].metric("Blocked", n_blocked)
    dash_cols[3].metric("Legitimate", n_legit)
    dash_cols[4].metric("Review", n_review)
    st.markdown("---")
    st.markdown("#### 🟢 Latest Transactions (refreshes every few seconds)")
    for tx in st.session_state.demo_transactions[:20]:
        status_map = {
            'pending': ('⏳', 'PENDING', '#f4f6f7'),
            'reviewing': ('🔎', 'REVIEWING', '#f9e79f'),
            'blocked': ('🚫', 'BLOCKED', '#fadbd8'),
            'approved': ('✅', 'LEGITIMATE', '#d5f5e3')
        }
        emoji, label, bgcolor = status_map.get(tx['status'], ('❓', tx['status'], '#d6dbdf'))
        st.markdown(f"""
        <div style="background-color:{bgcolor};border-left:4px solid #34495e;padding:8px;margin-bottom:10px;border-radius:4px;">
        <b>{tx['merchant']}</b> [{tx['card']}] <br/>
        <span style="color:#555;">Amount: ${tx['amount']:.2f} | Probability: {tx['fraud_prob']*100:.1f}%</span><br/>
        <span style="color:#888;">Time: {tx['datetime']}</span> <br/>
        <span style="font-size:18px;">{emoji}</span>
        <b style="color:#2d3436;padding-left:8px;">{label}</b>
        </div>
        """, unsafe_allow_html=True)
    st.caption("Dashboard demo is independent from model results and does not affect your uploaded analysis.")



