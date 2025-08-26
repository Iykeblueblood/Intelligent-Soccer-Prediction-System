import streamlit as st
import requests
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import joblib
import numpy as np
import os

# --- Configuration & Model Loading ---
def get_secret(section, key):
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        return None

GEMINI_API_KEY = get_secret("google", "gemini_api_key")
FOOTBALL_DATA_API_KEY = get_secret("footballdata", "api_key")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    GEMINI_ENABLED = True
else:
    GEMINI_ENABLED = False

@st.cache_resource
def load_ml_model(model_path):
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found. Please run 'python train_model.py' for this league."
    try:
        model = joblib.load(model_path)
        return model, f"Model '{model_path}' loaded."
    except Exception as e:
        return None, f"Error loading model: {e}"

@st.cache_data
def load_historical_data(file_path):
    if not os.path.exists(file_path):
        return None, f"Historical data file '{file_path}' not found."
    try:
        df = pd.read_csv(file_path)
        return df, "Historical data loaded."
    except Exception as e:
        return None, f"Error loading CSV: {e}"

@st.cache_data(ttl=3600)
def get_current_season_data(league_code):
    if not FOOTBALL_DATA_API_KEY: return None, None, "API key for football-data.org not found."
    headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
    teams_url = f"https://api.football-data.org/v4/competitions/{league_code}/teams"
    matches_url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=FINISHED"
    try:
        teams_data = requests.get(teams_url, headers=headers).json().get('teams', [])
        matches_data = requests.get(matches_url, headers=headers).json().get('matches', [])
        return teams_data, matches_data, "Success"
    except Exception as e:
        return None, None, f"Error fetching current data: {e}"

# --- Data Processing, H2H, and Engines ---
def process_team_stats(team_id, team_name, all_matches): # Now accepts team_name
    team_matches = [m for m in all_matches if (m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id)]
    team_matches.sort(key=lambda x: x['utcDate'], reverse=True)
    stats = {"form": "N/A", "form_string": "No recent matches found.", "avg_goals_scored": 0, "avg_goals_conceded": 0, "form_points": 0, "clean_sheets": 0, "failed_to_score": 0}
    if not team_matches: return stats
    
    form, form_string_list, goals_scored, goals_conceded, cs, fts = [], [], [], [], 0, 0
    
    for match in team_matches[:5]:
        score, winner, is_home = match['score'], match['score']['winner'], match['homeTeam']['id'] == team_id
        home_s, away_s = score['fullTime']['home'], score['fullTime']['away']
        if home_s is None or away_s is None: continue
        
        goals_for, goals_against = (home_s, away_s) if is_home else (away_s, home_s)
        if goals_for == 0: fts += 1
        if goals_against == 0: cs += 1
        goals_scored.append(goals_for); goals_conceded.append(goals_against)
        
        result = 'W' if (winner == 'HOME_TEAM' and is_home) or (winner == 'AWAY_TEAM' and not is_home) else 'D' if winner == 'DRAW' else 'L'
        form.append(result)

        # --- THIS IS THE BUG FIX ---
        # Correctly identify the opponent's name for the details string
        opponent_name = match['awayTeam']['name'] if is_home else match['homeTeam']['name']
        score_display = f"{goals_for}-{goals_against}"
        form_string_list.append(f"{result} ({score_display} vs {opponent_name})")
        # --- END OF BUG FIX ---

    stats.update({"form": "".join(form), "form_string": ", ".join(form_string_list), "avg_goals_scored": np.mean(goals_scored) if goals_scored else 0, "avg_goals_conceded": np.mean(goals_conceded) if goals_conceded else 0, "form_points": form.count('W')*3 + form.count('D'), "clean_sheets": cs, "failed_to_score": fts})
    return stats


# --- Definitive Team Name Mapping Dictionaries ---
PREMIER_LEAGUE_MAP = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Burnley FC": "Burnley", "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Liverpool FC": "Liverpool", "Luton Town FC": "Luton", "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nottingham Forest", "Sheffield United FC": "Sheff Utd",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves", "AFC Bournemouth": "Bournemouth"
}
LA_LIGA_MAP = {
    "Athletic Club": "Ath Bilbao", "Club Atlético de Madrid": "Ath Madrid", "CA Osasuna": "Osasuna",
    "Cádiz CF": "Cadiz", "Deportivo Alavés": "Alaves", "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe", "Girona FC": "Girona", "Granada CF": "Granada",
    "Rayo Vallecano de Madrid": "Rayo Vallecano", "RC Celta de Vigo": "Celta Vigo", "RCD Espanyol de Barcelona": "Espanyol",
    "RCD Mallorca": "Mallorca", "Real Betis Balompié": "Betis", "Real Madrid CF": "Real Madrid",
    "Real Sociedad de Fútbol": "Real Sociedad", "Sevilla FC": "Sevilla", "UD Almería": "Almeria",
    "UD Las Palmas": "Las Palmas", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal"
}
BUNDESLIGA_MAP = {
    "1. FC Heidenheim 1846": "Heidenheim", "1. FC Köln": "FC Koln", "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz", "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich", "RB Leipzig": "RB Leipzig",
    "SC Freiburg": "Freiburg", "SV Darmstadt 98": "Darmstadt", "TSG 1899 Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart", "VfL Bochum 1848": "VfL Bochum", "VfL Wolfsburg": "VfL Wolfsburg",
    "SV Werder Bremen": "Werder Bremen"
}

def get_h2h_history(team1_live_name, team2_live_name, historical_df, team_name_map):
    if historical_df is None: return "**Head-to-Head History:**\n\nHistorical data not available for this league."
    team1_hist_name = team_name_map.get(team1_live_name)
    team2_hist_name = team_name_map.get(team2_live_name)
    if not team1_hist_name or not team2_hist_name: return "**Head-to-Head History:**\n\nCould not reliably match team names to historical data. (Mapping missing)"
    h2h_matches = historical_df[((historical_df['home_team'] == team1_hist_name) & (historical_df['away_team'] == team2_hist_name)) | ((historical_df['home_team'] == team2_hist_name) & (historical_df['away_team'] == team1_hist_name))].copy()
    if h2h_matches.empty: return "**Head-to-Head History:**\n\nNo head-to-head meetings found in the last 2-3 seasons."
    h2h_matches['date'] = pd.to_datetime(h2h_matches['date'])
    h2h_matches = h2h_matches.sort_values(by='date', ascending=False)
    summary = ["**Recent Head-to-Head History (from past seasons):**"]
    for index, row in h2h_matches.head(5).iterrows():
        summary.append(f"- On {row['date'].strftime('%Y-%m-%d')}: {row['home_team']} {int(row['home_goals'])} - {int(row['away_goals'])} {row['away_team']}")
    return "\n".join(summary)

def run_aligned_prediction_engine(home_stats, away_stats, home_name, away_name):
    home_score = 1.5; reasons = [f"[+{1.5} pts Home] General Home Field Advantage."]
    form_diff = home_stats['form_points'] - away_stats['form_points']
    if form_diff > 6: home_score += 3; reasons.append(f"[+{3} pts Home] Overwhelming Form: {home_name} ({home_stats['form_points']} pts) vs {away_name} ({away_stats['form_points']} pts).")
    elif form_diff > 2: home_score += 1.5; reasons.append(f"[+{1.5} pts Home] Better Form: {home_name} ({home_stats['form_points']} pts) vs {away_name} ({away_stats['form_points']} pts).")
    elif form_diff < -6: home_score -= 3; reasons.append(f"[-{3} pts Home] Overwhelming Away Form: {away_name} ({away_stats['form_points']} pts) is in much better form.")
    elif form_diff < -2: home_score -= 1.5; reasons.append(f"[-{1.5} pts Home] Better Away Form: {away_name} ({away_stats['form_points']} pts) has better results.")
    off_matchup = home_stats['avg_goals_scored'] - away_stats['avg_goals_conceded']
    if off_matchup > 1.25: home_score += 2.5; reasons.append(f"[+{2.5} pts Home] Offensive Mismatch: {home_name}'s attack ({home_stats['avg_goals_scored']:.2f} gpg) should dominate.")
    elif off_matchup > 0.5: home_score += 1; reasons.append(f"[+{1} pt Home] Offensive Advantage.")
    def_matchup = away_stats['avg_goals_scored'] - home_stats['avg_goals_conceded']
    if def_matchup > 1.25: home_score -= 2.5; reasons.append(f"[-{2.5} pts Home] Defensive Mismatch: {away_name}'s attack ({away_stats['avg_goals_scored']:.2f} gpg) will be a major threat.")
    elif def_matchup > 0.5: home_score -= 1; reasons.append(f"[-{1} pt Home] Defensive Challenge.")
    if home_stats['avg_goals_scored'] > 2.2: home_score += 1; reasons.append(f"[+{1} pt Home] Prolific Home Attack.")
    if away_stats['avg_goals_scored'] > 2.0: home_score -= 1; reasons.append(f"[-{1} pt Home] Potent Away Attack.")
    if home_stats['avg_goals_conceded'] < 0.8: home_score += 1; reasons.append(f"[+{1} pt Home] Fortress Home Defense.")
    if away_stats['avg_goals_conceded'] < 1.0: home_score -= 1; reasons.append(f"[-{1} pt Home] Solid Away Defense.")
    if home_stats['clean_sheets'] >= 3: home_score += 1.5; reasons.append(f"[+{1.5} pts Home] Shutdown Home Defense ({home_stats['clean_sheets']}/5 clean sheets).")
    if away_stats['failed_to_score'] >=3: home_score += 1; reasons.append(f"[+{1} pt Home] Away Goal Drought ({away_stats['failed_to_score']}/5 matches with no goals).")
    if home_score > 4: prediction = f"High-Confidence Win for {home_name}"
    elif home_score > 1.5: prediction = f"Likely Win for {home_name}"
    elif home_score < -4: prediction = f"High-Confidence Win for {away_name}"
    elif home_score < -1.5: prediction = f"Likely Win for {away_name}"
    else: prediction = "Prediction: A Draw is the most likely outcome"
    return prediction, reasons, home_score

def get_ml_prediction(model, home_stats, away_stats):
    if model is None: return "ML Model not available.", None
    features = pd.DataFrame([[home_stats['avg_goals_scored'], away_stats['avg_goals_scored'], home_stats['avg_goals_conceded'], away_stats['avg_goals_conceded'], home_stats['form_points'], away_stats['form_points']]], columns=['h_avg_goals', 'a_avg_goals', 'h_avg_conceded', 'a_avg_conceded', 'h_form_pts', 'a_form_pts'])
    pred_code = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    pred_map = {1: "Home Win", 0: "Draw", 2: "Away Win"}
    return f"{pred_map[pred_code]} ({np.max(probs)*100:.1f}% confidence)", probs

# --- FINAL, CORRECTED GEMINI FUNCTION ---
def get_gemini_analysis(team1_name, team2_name, team1_data, team2_data, rb_prediction, ml_prediction, h2h_summary):
    if not GEMINI_ENABLED:
        return "Gemini API key not configured."

    # Use the detailed form string which includes scores and opponents
    t1_form_details = team1_data['form_string']
    t2_form_details = team2_data['form_string']

    prompt = f"""
    Act as an expert sports analyst. Analyze this upcoming football match with the following data:

    **Match & Team Data:**
    - Home Team: {team1_name}
    - Home Team Recent Form Details: {t1_form_details}
    - Away Team: {team2_name}
    - Away Team Recent Form Details: {t2_form_details}

    **Recent Head-to-Head History:**
    {h2h_summary}

    **System Predictions:**
    - Rule-Based Engine Prediction: "{rb_prediction}"
    - Machine Learning Model Prediction: "{ml_prediction}"

    **Your Task:**
    Provide a final, nuanced expert verdict (2-3 paragraphs).
    1.  Synthesize the predictions from the two systems.
    2.  **Analyze the details of the recent form.** Do not just look at Win/Loss. Look at the scorelines. A 5-0 win is more impressive than a 1-0 win. A narrow 1-2 loss to a strong team is better than a 0-4 loss to a weak team.
    3.  Incorporate the head-to-head history. Is there a psychological advantage or a clear pattern?
    4.  If the predictions disagree, explore why. Does the quality of recent form (e.g., a dominant win) contradict the overall ML model prediction?
    5.  Conclude with your definitive expert prediction and your most important reasoning.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("⚽ Intelligent Soccer Prediction System")
st.markdown("Using a **Point-Based Rule Engine**, a **ML Model**, and **Gemini**.")

LEAGUES = {
    "English Premier League": {"fd": "PL", "model_key": "premier_league", "csv": "premier_league_raw_data.csv", "map": PREMIER_LEAGUE_MAP},
    "Spanish La Liga": {"fd": "PD", "model_key": "la_liga", "csv": "la_liga_raw_data.csv", "map": LA_LIGA_MAP},
    "German Bundesliga": {"fd": "BL1", "model_key": "bundesliga", "csv": "bundesliga_raw_data.csv", "map": BUNDESLIGA_MAP},
}
st.sidebar.header("Match Selection")
selected_league_name = st.sidebar.selectbox("Select a League:", LEAGUES.keys())
league_config = LEAGUES[selected_league_name]
league_code = league_config["fd"]
model_path = f"trained_model_{league_config['model_key']}.pkl"
csv_path = league_config['csv']
team_name_map = league_config['map']

ml_model, ml_status = load_ml_model(model_path)
historical_df, historical_status = load_historical_data(csv_path)

if not FOOTBALL_DATA_API_KEY:
    st.error("Please review your API key configuration.")
else:
    teams, all_matches, status = get_current_season_data(league_code)
    if not teams:
        st.error(f"Could not fetch data for {selected_league_name}. Reason: {status}")
    else:
        teams_dict = {team['name']: team['id'] for team in teams}
        sorted_team_names = sorted(teams_dict.keys())
        team1_name = st.sidebar.selectbox("Select Home Team:", sorted_team_names, index=0)
        available_teams_for_t2 = [name for name in sorted_team_names if name != team1_name]
        team2_name = st.sidebar.selectbox("Select Away Team:", available_teams_for_t2, index=1)

        if st.sidebar.button("Analyze Match", type="primary"):
            team1_id, team2_id = teams_dict[team1_name], teams_dict[team2_name]
            with st.spinner('Performing advanced analysis...'):
                
                team1_stats = process_team_stats(team1_id, team1_name, all_matches)
                team2_stats = process_team_stats(team2_id, team2_name, all_matches)
                
                h2h_summary = get_h2h_history(team1_name, team2_name, historical_df, team_name_map)

                st.header(f"Analysis: {team1_name} vs. {team2_name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{team1_name} (Home)"); st.metric("Recent Form", team1_stats['form'])
                    with st.expander("View Details"): st.markdown(f"- {team1_stats['form_string'].replace(', ', '\n- ')}")
                with col2:
                    st.subheader(f"{team2_name} (Away)"); st.metric("Recent Form", team2_stats['form'])
                    with st.expander("View Details"): st.markdown(f"- {team2_stats['form_string'].replace(', ', '\n- ')}")
                
                st.markdown(h2h_summary)
                st.markdown("---")

                rb_prediction, rb_reasons, rb_score = run_aligned_prediction_engine(team1_stats, team2_stats, team1_name, team2_name)
                ml_prediction, ml_probs = get_ml_prediction(ml_model, team1_stats, team2_stats)

                st.subheader("📊 Prediction Summary")
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.markdown(f"##### 🤖 Rule-Based Engine (Score: {rb_score:.2f})")
                    st.success(rb_prediction)
                    with st.expander("Show Reasoning"): st.markdown('\n'.join(f'- {reason}' for reason in rb_reasons))
                with pred_col2:
                    st.markdown("##### 📈 Machine Learning Model")
                    st.success(ml_prediction)
                    if ml_probs is not None:
                        st.write("Probabilities:"); st.write(f"- {team1_name} Win: **{ml_probs[1]*100:.1f}%**\n- Draw: **{ml_probs[0]*100:.1f}%**\n- {team2_name} Win: **{ml_probs[2]*100:.1f}%**")

                if GEMINI_ENABLED:
                    st.markdown("---")
                    st.subheader("🧠 Final Expert Verdict")
                    with st.spinner("Gemini is synthesizing the results..."):
                        st.markdown(get_gemini_analysis(team1_name, team2_name, team1_stats, team2_stats, rb_prediction, ml_prediction, h2h_summary))

st.sidebar.markdown("---")
if not FOOTBALL_DATA_API_KEY: st.sidebar.warning("Football-data.org key missing.")
if not GEMINI_API_KEY: st.sidebar.warning("Gemini key missing.")
if ml_model is None: st.sidebar.error(f"ML Model Error for {selected_league_name}: {ml_status}")
if historical_df is None: st.sidebar.error(f"Historical Data Error for {selected_league_name}: {historical_status}")
