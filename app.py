import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 4px;
        color: #1a1a2e;
    }
    .section-sub {
        font-size: 13px;
        color: #6c757d;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653', '#A8DADC', '#1D3557']

# ── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = Path(__file__).parent / 'ipl_matches.csv'
    return pd.read_csv(path)

@st.cache_data
def train_models(df):
    le = LabelEncoder()
    df_ml = df.copy()
    for col in ['batting_first_team', 'bowling_first_team', 'venue', 'toss_decision', 'winner']:
        df_ml[col + '_enc'] = le.fit_transform(df_ml[col])
    df_ml['season_norm'] = (df_ml['season'] - df_ml['season'].min()) / (df_ml['season'].max() - df_ml['season'].min())

    features = ['batting_first_team_enc', 'bowling_first_team_enc', 'venue_enc', 'season_norm', 'toss_decision_enc']
    X = df_ml[features]
    y = df_ml['first_innings_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    df_ml['batting_first_wins'] = (df_ml['winner'] == df_ml['batting_first_team']).astype(int)
    features2 = features + ['first_innings_score']
    X2 = df_ml[features2]
    y2 = df_ml['batting_first_wins']
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X2_train, y2_train)
    clf_acc = accuracy_score(y2_test, clf.predict(X2_test))

    return rf, clf, rf_mae, rf_r2, clf_acc, df_ml, le, features

df = load_data()
rf_model, clf_model, rf_mae, rf_r2, clf_acc, df_ml, le, reg_features = train_models(df)

teams = sorted(df['batting_first_team'].unique())
venues = sorted(df['venue'].unique())

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## IPL Analyzer")
    st.markdown("---")
    page = st.radio("Navigate", ["Live Scores", "Dashboard", "Team Analysis", "Score Predictor", "ML Insights"])


# ══════════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ══════════════════════════════════════════════════════════════
# Live Scores Page
if page == "Live Scores":
    st.title("Live Cricket Scores")
    st.markdown("Real-time cricket match scores fetched directly in your browser.")
    st.markdown("---")

    import streamlit.components.v1 as components

    RAPIDAPI_KEY = "d66f3a7c9fmsh81d1ad00e4ae47ap190e24jsn27e5237bcaa9"

    live_html = f"""
    <div id="live-container" style="font-family: sans-serif;">
        <button onclick="fetchScores()" style="
            background:#457B9D; color:white; border:none;
            padding:10px 24px; border-radius:8px; font-size:15px;
            cursor:pointer; margin-bottom:20px;">
            Refresh Live Scores
        </button>
        <div id="scores-output">
            <p style="color:#6c757d;">Loading live scores...</p>
        </div>
    </div>

    <script>
    async function fetchScores() {{
        document.getElementById('scores-output').innerHTML = '<p style="color:#6c757d;">Fetching scores...</p>';
        try {{
            const response = await fetch('https://cricbuzz-cricket.p.rapidapi.com/matches/v1/live', {{
                method: 'GET',
                headers: {{
                    'x-rapidapi-host': 'cricbuzz-cricket.p.rapidapi.com',
                    'x-rapidapi-key': '{RAPIDAPI_KEY}'
                }}
            }});
            const data = await response.json();
            console.log('Cricbuzz Response:', data);

            // Cricbuzz returns typeMatches array
            let matches = [];
            if (data.typeMatches) {{
                data.typeMatches.forEach(type => {{
                    if (type.seriesMatches) {{
                        type.seriesMatches.forEach(series => {{
                            if (series.seriesAdWrapper && series.seriesAdWrapper.matches) {{
                                series.seriesAdWrapper.matches.forEach(m => {{
                                    if (m.matchInfo) matches.push({{
                                        team1: m.matchInfo.team1 ? m.matchInfo.team1.teamName : 'Team 1',
                                        team2: m.matchInfo.team2 ? m.matchInfo.team2.teamName : 'Team 2',
                                        status: m.matchInfo.status || 'Live',
                                        score: m.matchScore ? 
                                            (m.matchScore.team1Score ? 
                                                (m.matchScore.team1Score.inngs1 ? 
                                                    m.matchScore.team1Score.inngs1.runs + '/' + m.matchScore.team1Score.inngs1.wickets : 'Yet to bat')
                                                : 'Yet to bat') : 'Live',
                                        venue: m.matchInfo.venueInfo ? m.matchInfo.venueInfo.ground : '',
                                        series: m.matchInfo.seriesName || ''
                                    }});
                                }});
                            }}
                        }});
                    }}
                }});
            }}

            if (matches.length === 0) {{
                document.getElementById('scores-output').innerHTML = `
                    <div style="text-align:center; padding:60px; background:#f8f9fa; border-radius:12px;">
                        <h2 style="color:#457B9D;">No Live Matches Right Now</h2>
                        <p style="color:#6c757d; font-size:16px;">Live scores will appear here when a match is in progress.</p>
                        <p style="color:#6c757d; font-size:14px;">IPL matches are typically held at 3:30 PM and 7:30 PM IST.</p>
                    </div>`;
                return;
            }}

            let html = '';
            matches.forEach(match => {{
                const team1 = match.team1 || match.t1 || match.teamA || 'Team 1';
                const team2 = match.team2 || match.t2 || match.teamB || 'Team 2';
                const score = match.score || match.livescore || match.t1s || 'Live';
                const status = match.status || match.matchstatus || 'In Progress';
                const venue = match.venue || match.ground || match.stadium || '';

                html += `
                <div style="background:white; border:1px solid #e9ecef; border-radius:12px;
                            padding:20px; margin-bottom:16px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                    <h3 style="color:#1a1a2e; margin:0 0 12px 0;">
                        ${{team1}} <span style="color:#E63946;">vs</span> ${{team2}}
                    </h3>
                    <div style="display:flex; gap:20px; flex-wrap:wrap;">
                        <div style="background:#f8f9fa; padding:10px 16px; border-radius:8px;">
                            <div style="font-size:12px; color:#6c757d;">Status</div>
                            <div style="font-weight:600; color:#2A9D8F;">🔴 ${{status}}</div>
                        </div>
                        <div style="background:#f8f9fa; padding:10px 16px; border-radius:8px;">
                            <div style="font-size:12px; color:#6c757d;">Score</div>
                            <div style="font-weight:600; color:#1a1a2e;">${{score}}</div>
                        </div>
                        ${{venue ? `<div style="background:#f8f9fa; padding:10px 16px; border-radius:8px;">
                            <div style="font-size:12px; color:#6c757d;">Venue</div>
                            <div style="font-weight:600; color:#1a1a2e;">${{venue}}</div>
                        </div>` : ''}}
                    </div>
                </div>`;
            }});
            document.getElementById('scores-output').innerHTML = html;

        }} catch(err) {{
            document.getElementById('scores-output').innerHTML = `
                <div style="text-align:center; padding:60px; background:#f8f9fa; border-radius:12px;">
                    <h2 style="color:#457B9D;">No Live Matches Right Now</h2>
                    <p style="color:#6c757d; font-size:16px;">Live scores will appear here when a match is in progress.</p>
                    <p style="color:#6c757d; font-size:14px;">IPL matches are typically held at 3:30 PM and 7:30 PM IST.</p>
                </div>`;
        }}
    }}

    // Auto fetch on load
    fetchScores();

    // Auto refresh every 60 seconds
    setInterval(fetchScores, 60000);
    </script>
    """

    components.html(live_html, height=600, scrolling=True)

elif page == "Dashboard":
    st.title("IPL Match Analysis Dashboard")
    st.markdown("Complete analysis of IPL matches from 2015 to 2023")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(df))
    col2.metric("Avg 1st Innings", f"{df['first_innings_score'].mean():.0f} runs")
    col3.metric("Avg 2nd Innings", f"{df['second_innings_score'].mean():.0f} runs")
    toss_win_pct = (df['toss_winner'] == df['winner']).mean() * 100
    col4.metric("Toss Win Rate", f"{toss_win_pct:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Total Wins by Team</div>', unsafe_allow_html=True)
        wins = df['winner'].value_counts().reset_index()
        wins.columns = ['Team', 'Wins']
        fig = px.bar(wins, x='Team', y='Wins', color='Team',
                     color_discrete_sequence=COLORS, text='Wins')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, xaxis_tickangle=-30,
                          plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['first_innings_score'], name='1st Innings',
                                   marker_color='#E63946', opacity=0.75, nbinsx=25))
        fig.add_trace(go.Histogram(x=df['second_innings_score'], name='2nd Innings',
                                   marker_color='#457B9D', opacity=0.75, nbinsx=25))
        fig.update_layout(barmode='overlay', plot_bgcolor='white', paper_bgcolor='white',
                          legend=dict(x=0.7, y=0.95), margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Average Scores by Season</div>', unsafe_allow_html=True)
        season_avg = df.groupby('season')[['first_innings_score', 'second_innings_score']].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=season_avg['season'], y=season_avg['first_innings_score'],
                                 mode='lines+markers', name='1st Innings',
                                 line=dict(color='#E63946', width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=season_avg['season'], y=season_avg['second_innings_score'],
                                 mode='lines+markers', name='2nd Innings',
                                 line=dict(color='#457B9D', width=2.5), marker=dict(size=8)))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          xaxis_title='Season', yaxis_title='Avg Score',
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_d:
        st.markdown('<div class="section-title">Toss Decision Impact</div>', unsafe_allow_html=True)
        toss_win = (df['toss_winner'] == df['winner']).sum()
        toss_lose = len(df) - toss_win
        fig = px.pie(values=[toss_win, toss_lose],
                     names=['Toss winner won', 'Toss winner lost'],
                     color_discrete_sequence=['#2A9D8F', '#E9C46A'],
                     hole=0.35)
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Venue x Season: Avg First Innings Score</div>', unsafe_allow_html=True)
    top_venues = df['venue'].value_counts().head(6).index
    heat_data = df[df['venue'].isin(top_venues)].groupby(
        ['venue', 'season'])['first_innings_score'].mean().unstack(fill_value=0)
    heat_data.index = [v.replace(' Stadium', '').replace(' Cricket Association', '') for v in heat_data.index]
    fig = px.imshow(heat_data, color_continuous_scale='YlOrRd', aspect='auto', text_auto='.0f')
    fig.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE 2: Team Analysis
# ══════════════════════════════════════════════════════════════
elif page == "Team Analysis":
    st.title("Team Deep Dive")
    selected_team = st.selectbox("Select a team to analyze", teams)

    team_df = df[(df['team1'] == selected_team) | (df['team2'] == selected_team)]
    team_wins = df[df['winner'] == selected_team]
    team_batting = df[df['batting_first_team'] == selected_team]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(team_df))
    col2.metric("Total Wins", len(team_wins))
    col3.metric("Win Rate", f"{len(team_wins)/len(team_df)*100:.1f}%")
    col4.metric("Avg Score (Bat 1st)", f"{team_batting['first_innings_score'].mean():.0f}" if len(team_batting) else "N/A")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Wins by Season</div>', unsafe_allow_html=True)
        wins_by_season = team_wins.groupby('season').size().reset_index(name='Wins')
        fig = px.bar(wins_by_season, x='season', y='Wins', color_discrete_sequence=['#E63946'])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          xaxis_title='Season', margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">Score Distribution when Batting First</div>', unsafe_allow_html=True)
        if len(team_batting) > 0:
            fig = px.histogram(team_batting, x='first_innings_score',
                               color_discrete_sequence=['#457B9D'], nbins=20)
            fig.add_vline(x=team_batting['first_innings_score'].mean(),
                          line_dash='dash', line_color='#E63946',
                          annotation_text=f"Mean: {team_batting['first_innings_score'].mean():.0f}")
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(t=20, b=10))
            st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Win/Loss vs Each Opponent</div>', unsafe_allow_html=True)
    opponents = [t for t in teams if t != selected_team]
    h2h_data = []
    for opp in opponents:
        matches = df[((df['team1'] == selected_team) & (df['team2'] == opp)) |
                     ((df['team1'] == opp) & (df['team2'] == selected_team))]
        wins_count = (matches['winner'] == selected_team).sum()
        losses_count = len(matches) - wins_count
        h2h_data.append({'Opponent': opp.replace(' ', '\n'), 'Wins': wins_count, 'Losses': losses_count})
    h2h_df = pd.DataFrame(h2h_data)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Wins', x=h2h_df['Opponent'], y=h2h_df['Wins'], marker_color='#2A9D8F'))
    fig.add_trace(go.Bar(name='Losses', x=h2h_df['Opponent'], y=h2h_df['Losses'], marker_color='#E63946'))
    fig.update_layout(barmode='group', plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(t=20, b=10))
    st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
# PAGE 3: Score Predictor
# ══════════════════════════════════════════════════════════════
elif page == "Score Predictor":
    st.title("Match Score and Winner Predictor")
    st.markdown("Select match details below to predict the first innings score and likely winner.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting First Team", teams)
        venue = st.selectbox("Venue", venues)
    with col2:
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
        season = st.slider("Season", 2015, 2023, 2023)
        toss_dec = st.radio("Toss Decision", ['bat', 'field'], horizontal=True)

    if st.button("Predict Now", use_container_width=True):
        le2 = LabelEncoder()
        df_enc = df.copy()
        for col in ['batting_first_team', 'bowling_first_team', 'venue', 'toss_decision']:
            le2.fit(df_enc[col])
            df_enc[col + '_enc'] = le2.transform(df_enc[col])

        row = pd.DataFrame([{
            'batting_first_team_enc': df_enc[df_enc['batting_first_team'] == batting_team]['batting_first_team_enc'].iloc[0] if batting_team in df_enc['batting_first_team'].values else 0,
            'bowling_first_team_enc': df_enc[df_enc['bowling_first_team'] == bowling_team]['bowling_first_team_enc'].iloc[0] if bowling_team in df_enc['bowling_first_team'].values else 0,
            'venue_enc': df_enc[df_enc['venue'] == venue]['venue_enc'].iloc[0] if venue in df_enc['venue'].values else 0,
            'season_norm': (season - 2015) / (2023 - 2015),
            'toss_decision_enc': 0 if toss_dec == 'bat' else 1,
        }])

        predicted_score = int(rf_model.predict(row)[0])

        row2 = row.copy()
        row2['first_innings_score'] = predicted_score

        # Safe win probability extraction
        win_prob = clf_model.predict_proba(row2)[0]
        classes = list(clf_model.classes_)
        if 1 in classes:
            batting_win_pct = win_prob[classes.index(1)] * 100
        else:
            batting_win_pct = 0.0
        bowling_win_pct = 100 - batting_win_pct

        st.markdown("---")
        st.markdown("### Prediction Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Score", f"{predicted_score} runs", f"+-{rf_mae:.0f} runs")
        c2.metric(f"{batting_team} Win %", f"{batting_win_pct:.0f}%")
        c3.metric(f"{bowling_team} Win %", f"{bowling_win_pct:.0f}%")

        fig = go.Figure(go.Bar(
            x=[batting_team, bowling_team],
            y=[batting_win_pct, bowling_win_pct],
            marker_color=['#2A9D8F', '#E63946'],
            text=[f"{batting_win_pct:.0f}%", f"{bowling_win_pct:.0f}%"],
            textposition='outside'
        ))
        fig.update_layout(
            title="Win Probability",
            plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(range=[0, 110], title='Win %'),
            showlegend=False, margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig, width='stretch')

        hist = df[df['batting_first_team'] == batting_team]['first_innings_score']
        st.markdown(f"**Historical context for {batting_team} batting first:**")
        st.markdown(f"Average: **{hist.mean():.0f}** | Best: **{hist.max()}** | Worst: **{hist.min()}** runs")

# ══════════════════════════════════════════════════════════════
# PAGE 4: ML Insights
# ══════════════════════════════════════════════════════════════
elif page == "ML Insights":
    st.title("ML Model Insights")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score Predictor MAE", f"+-{rf_mae:.1f} runs")
    col2.metric("Score Predictor R2", f"{rf_r2:.3f}")
    col3.metric("Win Predictor Accuracy", f"{clf_acc*100:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Feature Importance (Score Predictor)</div>', unsafe_allow_html=True)
        feat_names = ['Batting Team', 'Bowling Team', 'Venue', 'Season', 'Toss Decision']
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance')
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-title">ANOVA Test - Team Score Differences</div>', unsafe_allow_html=True)
        team_groups = [df[df['batting_first_team'] == t]['first_innings_score'].values for t in teams]
        f_stat, p_value = stats.f_oneway(*team_groups)
        corr_r, corr_p = stats.pearsonr(df['first_innings_score'], df['second_innings_score'])

        st.markdown(f"""
| Stat | Value |
|---|---|
| ANOVA F-statistic | {f_stat:.4f} |
| ANOVA p-value | {p_value:.4f} |
| Result | {'Significant difference' if p_value < 0.05 else 'No significant difference'} |
| Pearson r (1st vs 2nd) | {corr_r:.4f} |
| Pearson p-value | {corr_p:.4f} |
        """)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ['first_innings_score', 'first_innings_wickets', 'second_innings_score',
                'second_innings_wickets', 'win_by_runs', 'win_by_wickets',
                'player_of_match_runs', 'player_of_match_wickets', 'season']
    corr_matrix = df[num_cols].corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                    text_auto='.2f', aspect='auto')
    fig.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig, width='stretch')
