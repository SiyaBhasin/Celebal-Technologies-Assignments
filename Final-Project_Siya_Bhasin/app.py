"""
app.py  —  X Education Lead Scoring CRM
Celebal Technologies Internship | Week 9 | NTCC Project
Run:  python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="X Education | Lead Scoring CRM",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e2535 0%, #252d3d 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin: 4px 0;
    }
    .kpi-label  { font-size: 12px; color: #8892a4; letter-spacing: 0.5px; margin-bottom: 6px; text-transform: uppercase; }
    .kpi-value  { font-size: 30px; font-weight: 700; color: #ffffff; }
    .kpi-delta  { font-size: 12px; margin-top: 4px; }
    .kpi-green  { color: #48bb78; }
    .kpi-red    { color: #fc8181; }
    .kpi-blue   { color: #63b3ed; }
    .kpi-yellow { color: #f6e05e; }
    .score-box {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        border: 2px solid #2d3748;
    }
    .score-number { font-size: 80px; font-weight: 800; line-height: 1; }
    .badge-hot   { background:#276749; color:#9ae6b4; padding:4px 14px; border-radius:20px; font-weight:600; font-size:15px; }
    .badge-warm  { background:#744210; color:#fbd38d; padding:4px 14px; border-radius:20px; font-weight:600; font-size:15px; }
    .badge-cold  { background:#2a4365; color:#90cdf4; padding:4px 14px; border-radius:20px; font-weight:600; font-size:15px; }
    .badge-vcold { background:#44337a; color:#d6bcfa; padding:4px 14px; border-radius:20px; font-weight:600; font-size:15px; }
    .section-divider { border-top: 1px solid #2d3748; margin: 20px 0; }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0);
        color: white; border: none; border-radius: 8px;
        padding: 10px 32px; font-size: 15px; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #4299e1, #3182ce); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD MODEL + DATA
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('lead_scoring_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('leads_scored.csv')

try:
    bundle = load_model()
    df     = load_data()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 X Education")
    st.markdown("**Lead Scoring CRM Platform**")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["📊 Executive Dashboard", "📋 Sales Lead List", "🔮 Live Lead Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    if MODEL_LOADED:
        st.markdown("### Model Info")
        st.success("✅ Models Loaded")
        st.markdown(f"""
        - **Ensemble** (LR + RF + GB)
        - **AUC Score:** `{bundle['ensemble_auc']:.4f}`
        - **Accuracy:** `{bundle['ensemble_acc']:.4f}`
        - **Threshold:** `{bundle['optimal_threshold']:.2f}`
        """)
    else:
        st.error("⚠️ Run notebook save-model cell first")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.caption("Celebal Technologies Internship\nWeek 9 | NTCC Project")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def kpi_card(label, value, delta=None, color='blue'):
    delta_html = f'<div class="kpi-delta kpi-{color}">{delta}</div>' if delta else ''
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}</div>"""

def category_badge(cat):
    cls = {'Hot Lead':'badge-hot','Warm Lead':'badge-warm',
           'Cold Lead':'badge-cold','Very Cold Lead':'badge-vcold'}.get(cat,'badge-cold')
    return f'<span class="{cls}">{cat}</span>'

def score_color(score):
    if score >= 75:  return '#48bb78'
    elif score >= 50: return '#f6ad55'
    elif score >= 25: return '#63b3ed'
    else:             return '#b794f4'

def categorize(s):
    if s >= 75:  return '🟢 Hot Lead'
    elif s >= 50: return '🟠 Warm Lead'
    elif s >= 25: return '🔵 Cold Lead'
    else:         return '🔴 Very Cold Lead'

if not MODEL_LOADED:
    st.title("🎯 X Education Lead Scoring CRM")
    st.error("Run the **Save Model** cell in your notebook first, then restart the app.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("Real-time overview of X Education's lead pipeline and model performance.")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    total     = len(df)
    hot       = (df['Lead_Category'] == 'Hot Lead').sum()
    avg_score = df['Lead_Score'].mean()
    hot_pct   = hot / total * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("TOTAL LEADS", f"{total:,}", "Full pipeline"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("HOT LEADS 🟢", f"{hot:,}", f"{hot_pct:.1f}% of pipeline", "green"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("AVG LEAD SCORE", f"{avg_score:.1f}", "Out of 100", "blue"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("MODEL ROC-AUC", f"{bundle['ensemble_auc']:.3f}", "Ensemble LR+RF+GB", "blue"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Business Impact: Before vs After Model")
    b1, b2, b3, b4 = st.columns(4)
    calls_saved = int(total * (1 - hot_pct/100))
    with b1: st.markdown(kpi_card("BASELINE CONVERSION", "~30%", "Without model", "red"), unsafe_allow_html=True)
    with b2: st.markdown(kpi_card("MODEL TARGET PRECISION", "~80%", "↑ 166% improvement", "green"), unsafe_allow_html=True)
    with b3: st.markdown(kpi_card("OPTIMAL THRESHOLD", f"{bundle['optimal_threshold']:.2f}", "Ensemble score cutoff", "yellow"), unsafe_allow_html=True)
    with b4: st.markdown(kpi_card("WASTED CALLS AVOIDED", f"{calls_saved:,}", "vs calling everyone", "green"), unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Charts using Streamlit native ──────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Lead Category Breakdown")
        cat_counts = df['Lead_Category'].value_counts()
        order = ['Hot Lead','Warm Lead','Cold Lead','Very Cold Lead']
        cat_df = pd.DataFrame({
            'Category': order,
            'Count': [cat_counts.get(c, 0) for c in order]
        }).set_index('Category')
        st.bar_chart(cat_df, color='#3182ce')

    with col_right:
        st.markdown("#### Lead Score Distribution")
        score_bins = pd.cut(df['Lead_Score'], bins=[0,25,50,75,100],
                            labels=['0-25 (Very Cold)','25-50 (Cold)','50-75 (Warm)','75-100 (Hot)'])
        score_dist = score_bins.value_counts().sort_index()
        score_df = pd.DataFrame({'Leads': score_dist.values}, index=score_dist.index)
        st.bar_chart(score_df, color='#48bb78')

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Feature Drivers table ───────────────────────────────────────────────
    st.markdown("#### 🔍 Top Conversion Drivers (Logistic Regression Coefficients)")
    coef = bundle['coef_series']
    top_coef = pd.DataFrame({
        'Feature': coef.abs().sort_values(ascending=False).head(12).index,
        'Coefficient': coef.reindex(coef.abs().sort_values(ascending=False).head(12).index).values,
    })
    top_coef['Direction'] = top_coef['Coefficient'].apply(
        lambda x: '🟢 Increases conversion' if x > 0 else '🔴 Decreases conversion'
    )
    top_coef['Strength'] = top_coef['Coefficient'].abs().round(3)
    top_coef = top_coef[['Feature','Direction','Strength']].reset_index(drop=True)
    st.dataframe(top_coef, use_container_width=True, hide_index=True)

    st.info("""
    **CEO Insight:**
    Leads from **Landing Page Submission** who spent significant **time on website**
    are your hottest prospects. Leads via Olark Chat with **0 time on site** almost never convert.
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SALES LEAD LIST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Sales Lead List":
    total = len(df)  # needed for KPI cards
    st.markdown("## 📋 Sales Lead List")
    st.markdown("Filter and sort your lead pipeline. Prioritise who to call first.")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        min_score = st.slider("Minimum Lead Score", 0, 100,
                              int(bundle['optimal_threshold'] * 100), 5)
    with f2:
        sel_cat = st.selectbox("Lead Category",
                               ['All','Hot Lead','Warm Lead','Cold Lead','Very Cold Lead'])
    with f3:
        sources = ['All'] + sorted(df['Lead Source'].dropna().unique().tolist()) \
                  if 'Lead Source' in df.columns else ['All']
        sel_src = st.selectbox("Lead Source", sources)

    filtered = df[df['Lead_Score'] >= min_score].copy()
    if sel_cat != 'All':
        filtered = filtered[filtered['Lead_Category'] == sel_cat]
    if sel_src != 'All' and 'Lead Source' in filtered.columns:
        filtered = filtered[filtered['Lead Source'] == sel_src]

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_card("LEADS IN VIEW", f"{len(filtered):,}", f"of {total:,} total"), unsafe_allow_html=True)
    with k2:
        hot_f = (filtered['Lead_Category'] == 'Hot Lead').sum()
        st.markdown(kpi_card("HOT LEADS", f"{hot_f:,}", "Priority contacts", "green"), unsafe_allow_html=True)
    with k3:
        avg_f = filtered['Lead_Score'].mean() if len(filtered) else 0
        st.markdown(kpi_card("AVG SCORE", f"{avg_f:.1f}", "Filtered set", "blue"), unsafe_allow_html=True)
    with k4:
        if 'Converted' in filtered.columns and len(filtered):
            cr = filtered['Converted'].mean() * 100
            st.markdown(kpi_card("ACTUAL CONV RATE", f"{cr:.1f}%", "In filtered set", "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### Showing **{len(filtered):,}** leads — sorted by Lead Score ↓")

    display_cols = ['Priority_Rank','Lead_Score','Lead_Category','Lead Origin',
                    'Lead Source','Total Time Spent on Website',
                    'TotalVisits','Last Activity','Converted']
    display_cols = [c for c in display_cols if c in filtered.columns]

    show_df = filtered[display_cols].head(500).reset_index(drop=True)

    def color_score(val):
        if not isinstance(val, (int, float)): return ''
        if val >= 75:   return 'background-color: #276749; color: #9ae6b4'
        elif val >= 50: return 'background-color: #7b341e; color: #fbd38d'
        elif val >= 25: return 'background-color: #2a4365; color: #90cdf4'
        else:           return 'background-color: #44337a; color: #d6bcfa'

    st.dataframe(
        show_df.style.map(color_score, subset=['Lead_Score']),
        use_container_width=True, height=480
    )

    csv = filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Leads as CSV", csv,
                       'filtered_leads.csv', 'text/csv')


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — LIVE LEAD PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Lead Predictor":
    st.markdown("## 🔮 Live Lead Predictor")
    st.markdown("Simulate a **brand-new prospect** and instantly score them.")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📝 Enter Prospect Behaviour")
        lead_origin = st.selectbox("Lead Origin",
            ['Landing Page Submission','API','Lead Add Form','Lead Import','Quick Add Form'])
        lead_source = st.selectbox("Lead Source",
            ['Google','Direct Traffic','Organic Search','Olark Chat',
             'Referral Sites','Facebook','WeLearn','Social Media'])
        last_activity = st.selectbox("Last Activity",
            ['Email Opened','SMS Sent','Olark Chat Conversation',
             'Page Visited on Website','Email Bounced','Email Link Clicked',
             'Unreachable','Converted to Lead','Form Submitted on Website'])
        occupation = st.selectbox("Current Occupation",
            ['Unemployed','Working Professional','Student','Businessman','Housewife','Other'])
        st.markdown("---")
        time_on_site = st.slider("Total Time Spent on Website (seconds)", 0, 2500, 800, 50)
        total_visits = st.slider("Total Visits", 0, 30, 4, 1)
        page_views   = st.slider("Page Views Per Visit", 0.0, 20.0, 3.0, 0.5)
        do_not_email = st.checkbox("Do Not Email", value=False)
        do_not_call  = st.checkbox("Do Not Call",  value=False)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn  = st.button("🎯 Calculate Lead Score", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Scoring Result")

        if predict_btn:
            all_features = bundle['all_features']
            input_vec    = pd.DataFrame(0, index=[0], columns=all_features)

            num_map = {
                'TotalVisits': total_visits,
                'Total Time Spent on Website': time_on_site,
                'Page Views Per Visit': page_views,
            }
            for feat, val in num_map.items():
                if feat in input_vec.columns:
                    input_vec[feat] = val

            if 'Do Not Email' in input_vec.columns:
                input_vec['Do Not Email'] = 1 if do_not_email else 0
            if 'Do Not Call' in input_vec.columns:
                input_vec['Do Not Call']  = 1 if do_not_call  else 0

            def set_ohe(prefix, value):
                col = f"{prefix}_{value}"
                if col in input_vec.columns:
                    input_vec[col] = 1

            set_ohe('Lead Origin', lead_origin)
            set_ohe('Lead Source', lead_source)
            set_ohe('Last Activity', last_activity)
            set_ohe('What is your current occupation', occupation)

            scaler    = bundle['scaler']
            num_feats = bundle['num_features']
            selected  = bundle['selected_features']

            input_scaled = input_vec.copy()
            valid_num    = [f for f in num_feats if f in input_scaled.columns]
            input_scaled[valid_num] = scaler.transform(input_scaled[valid_num])

            available = [f for f in selected if f in input_scaled.columns]
            input_sel = input_scaled[available]

            lr_p = bundle['lr_model'].predict_proba(input_sel)[0, 1]
            rf_p = bundle['rf_model'].predict_proba(input_sel)[0, 1]
            gb_p = bundle['gb_model'].predict_proba(input_sel)[0, 1]
            ens  = (lr_p + rf_p + gb_p) / 3

            score    = round(ens * 100, 1)
            category = ('Hot Lead' if score >= 75 else 'Warm Lead' if score >= 50
                        else 'Cold Lead' if score >= 25 else 'Very Cold Lead')
            color    = score_color(score)
            above_t  = score >= bundle['optimal_threshold'] * 100

            # Score display
            st.markdown(f"""
            <div class="score-box">
                <div style="color:#8892a4;font-size:13px;letter-spacing:1px;margin-bottom:12px;">
                    CALCULATED LEAD SCORE
                </div>
                <div class="score-number" style="color:{color};">{score}</div>
                <div style="color:#8892a4;font-size:14px;margin-top:4px;">out of 100</div>
                <div style="margin-top:16px;">{category_badge(category)}</div>
                <div style="color:#8892a4;font-size:13px;margin-top:14px;">
                    Estimated Conversion Probability:
                    <strong style="color:{color};">{ens*100:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if category == 'Hot Lead':
                st.success(f"""
                🟢 **HOT LEAD — Immediate Action Required**
                Conversion probability: **{ens*100:.1f}%**
                - 📞 Assign to senior sales rep **immediately**
                - 📱 Send SMS follow-up within 30 minutes
                - 📧 Send personalised email with course brochure
                """)
            elif category == 'Warm Lead':
                st.warning(f"""
                🟠 **WARM LEAD — Follow Up Within 24 Hours**
                Conversion probability: **{ens*100:.1f}%**
                - 📧 Send nurture email sequence
                - 📞 Schedule call in 1–2 days
                """)
            elif category == 'Cold Lead':
                st.info(f"""
                🔵 **COLD LEAD — Low Priority**
                Conversion probability: **{ens*100:.1f}%**
                - 📧 Add to weekly newsletter
                - ⏳ Re-evaluate after 2 weeks
                """)
            else:
                st.error(f"""
                🔴 **VERY COLD LEAD — Do Not Contact**
                Conversion probability: **{ens*100:.1f}%**
                - 🚫 Remove from active sales queue
                """)

            st.markdown("#### Model Breakdown")
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(kpi_card("Logistic Reg", f"{lr_p*100:.1f}", "Interpretable"), unsafe_allow_html=True)
            with m2: st.markdown(kpi_card("Random Forest", f"{rf_p*100:.1f}", "Ensemble"), unsafe_allow_html=True)
            with m3: st.markdown(kpi_card("Grad Boosting", f"{gb_p*100:.1f}", "Boosted"), unsafe_allow_html=True)

            threshold_score = bundle['optimal_threshold'] * 100
            st.markdown(f"""
            <br>
            <div style="background:#1e2535;border-radius:10px;padding:14px 20px;border:1px solid #2d3748;">
                Optimal threshold: <strong style="color:#f6e05e;">{threshold_score:.0f}/100</strong>
                &nbsp;|&nbsp; This lead scores <strong style="color:{color};">{score}</strong>
                &nbsp;→&nbsp;
                <strong style="color:{'#48bb78' if above_t else '#fc8181'};">
                    {'✅ Above threshold — Contact this lead' if above_t else '❌ Below threshold — Skip'}
                </strong>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#1e2535;border-radius:16px;padding:60px 32px;
                        text-align:center;border:2px dashed #2d3748;">
                <div style="font-size:56px;margin-bottom:16px;">🎯</div>
                <div style="color:#8892a4;font-size:16px;line-height:1.8;">
                    Fill in the prospect details on the left<br>
                    and click <strong style="color:white;">Calculate Lead Score</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 Score Reference")
            st.markdown("""
            | Score | Category | Action |
            |---|---|---|
            | 75–100 | 🟢 Hot Lead | Immediate contact |
            | 50–74  | 🟠 Warm Lead | Follow up in 24h |
            | 25–49  | 🔵 Cold Lead | Nurture campaign |
            | 0–24   | 🔴 Very Cold | Do not contact |
            """)
