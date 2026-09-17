"""Dashboard Streamlit : funnel, importance, attribution et scoring en direct.

Lancement : streamlit run app/dashboard.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import make_synthetic, funnel_from_hits  # noqa: E402
import joblib  # noqa: E402

st.set_page_config(page_title='GA Journey Intelligence', layout='wide')
st.title('Customer Journey Intelligence')
st.caption('Parcours clients, propension d\'achat et attribution sur donnees Google Analytics')

METRICS = ROOT / 'models' / 'metrics.json'
report = json.loads(METRICS.read_text(encoding='utf-8')) if METRICS.exists() else None

tab1, tab2, tab5, tab3, tab4 = st.tabs(
    ['Funnel', 'Modeles', 'Ciblage & profit', 'Attribution', 'Scoring en direct'])

with tab1:
    st.subheader('Tunnel de conversion (echantillon synthetique)')
    s, h = make_synthetic(n_customers=6000, seed=7)
    f = funnel_from_hits(h)
    f['taux_vs_precedent_%'] = (100 * f['sessions'] / f['sessions'].shift(1)).round(1)
    st.dataframe(f, use_container_width=True)
    st.bar_chart(f.set_index('etape')['sessions'])

with tab2:
    if report:
        for name in ('insession', 'customer'):
            st.subheader(f'Modele {name}')
            m = report[name]['metrics']
            c = st.columns(4)
            c[0].metric('ROC-AUC', m['roc_auc'])
            c[1].metric('PR-AUC', m['pr_auc'])
            c[2].metric('Top decile capte', f"{m['top_decile_capture_%']}%")
            c[3].metric('Top 30% capte', f"{m['top30_capture_%']}%")
            imp = pd.Series(report[name]['importance']).sort_values()
            st.bar_chart(imp)
    else:
        st.info('Lance d\'abord : python -m src.pipeline train')

with tab3:
    if report and 'attribution' in report:
        a = report['attribution']
        st.subheader('Attribution Markov (removal effect)')
        st.write(f"Probabilite de conversion de base : {a['base_conversion_prob']} | conversions : {a['n_conversions']}")
        rows = [{'canal': k, 'conversions_attribuees': v['attributed_conversions'],
                 'removal_effect': v['removal_effect']} for k, v in a['channels'].items()]
        dfa = pd.DataFrame(rows)
        st.dataframe(dfa, use_container_width=True)
        st.bar_chart(dfa.set_index('canal')['conversions_attribuees'])
    else:
        st.info('Attribution absente : lance l\'entrainement.')

with tab5:
    st.subheader('Ciblage et profit : combien contacter pour maximiser le gain net')
    scores_path = ROOT / 'models' / 'scores_customer_test.csv'
    if not scores_path.exists():
        st.info('Scores absents : lance `python scripts/make_figures.py` pour les generer.')
    else:
        sc = pd.read_csv(scores_path)
        y = sc['y_true'].to_numpy()
        p = sc['proba'].to_numpy()
        st.caption('Modele de production (cross-session), sur le jeu de test reel. '
                   'Regle la valeur d\'un acheteur et le cout d\'une action : le simulateur '
                   'trouve combien de trafic cibler pour maximiser le profit net.')
        cc = st.columns(2)
        value_tp = cc[0].slider('Gain par acheteur capte (EUR)', 10, 500, 100, 10)
        cost_action = cc[1].slider('Cout d\'une action marketing par session ciblee (EUR)', 1, 50, 5, 1)

        order = np.argsort(-p)
        ys = y[order]
        tp = np.cumsum(ys)
        n = np.arange(1, len(ys) + 1)
        profit = value_tp * tp - cost_action * n
        frac = 100.0 * n / len(ys)
        k = int(np.argmax(profit))

        m = st.columns(4)
        m[0].metric('Trafic a cibler', f'{100*(k+1)/len(ys):.0f}%')
        m[1].metric('Acheteurs captes', f'{100*tp[k]/max(y.sum(),1):.0f}%')
        m[2].metric('Profit net maximal', f'{profit[k]:,.0f} EUR'.replace(',', ' '))
        m[3].metric('Lift vs aleatoire', f'{(tp[k]/(k+1))/max(y.mean(),1e-9):.1f}x')

        idx = np.linspace(0, len(ys) - 1, min(len(ys), 400)).astype(int)
        curve = pd.DataFrame({'profit_net_EUR': profit[idx]},
                             index=np.round(frac[idx], 2))
        curve.index.name = '% du trafic cible (mieux scores en premier)'
        st.line_chart(curve)
        st.caption(f'Optimum : cibler les {100*(k+1)/len(ys):.0f}% mieux scores capte '
                   f'{100*tp[k]/max(y.sum(),1):.0f}% des acheteurs, pour un profit net de '
                   f'{profit[k]:,.0f} EUR sur le jeu de test.'.replace(',', ' '))

with tab4:
    st.subheader('Scorer une session')
    art_path = ROOT / 'models' / 'customer.joblib'
    if not art_path.exists():
        st.info('Modele absent : lance l\'entrainement.')
    else:
        art = joblib.load(art_path)
        col = st.columns(3)
        channel = col[0].selectbox('Canal', ['Organic Search', 'Direct', 'Referral', 'Paid Search', 'Social', 'Display', 'Affiliates'])
        device = col[1].selectbox('Appareil', ['desktop', 'mobile', 'tablet'])
        country = col[2].selectbox('Pays', ['United States', 'India', 'United Kingdom', 'Canada', 'France', 'Germany', 'Brazil', 'Japan', 'Other'])
        visit_number = col[0].number_input('Numero de visite', 1, 50, 3)
        prior_purchases = col[1].number_input('Achats passes', 0, 20, 1)
        days_since_last = col[2].number_input('Jours depuis derniere visite', -1, 365, 5)
        feats = art['features']
        row = {f: 0 for f in feats}
        row.update({'channelGrouping': channel, 'source': channel, 'medium': 'referral',
                    'device_category': device, 'os': 'Windows', 'is_mobile': int(device == 'mobile'),
                    'country': country, 'sub_continent': 'Northern America',
                    'visit_number': visit_number, 'new_visit': int(visit_number == 1), 'hour': 20,
                    'prior_sessions': max(visit_number - 1, 0), 'prior_purchases': prior_purchases,
                    'prior_pageviews_avg': 5.0, 'ever_purchased_before': int(prior_purchases > 0),
                    'days_since_last': days_since_last, 'is_returning': int(visit_number > 1)})
        df = pd.DataFrame([row], columns=feats)
        for cc in art['cat']:
            df[cc] = pd.Categorical(df[cc].astype(str), categories=art['categories'][cc])
        for cc in [f for f in feats if f not in art['cat']]:
            df[cc] = pd.to_numeric(df[cc], errors='coerce').fillna(0)
        p = float(art['model'].predict_proba(df)[:, 1][0])
        st.metric('Probabilite d\'achat estimee', f'{p:.1%}')
