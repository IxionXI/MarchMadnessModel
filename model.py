import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import json
import os
import warnings

# Hide messy warnings
warnings.filterwarnings("ignore") 

# ==========================================
# PHASE 1: LOAD AND PREPARE HISTORICAL DATA
# ==========================================
print("Loading historical data and calculating 10 advanced metrics...")

results = pd.read_csv('MNCAATourneyCompactResults.csv')
seeds = pd.read_csv('MNCAATourneySeeds.csv')
rankings = pd.read_csv('MMasseyOrdinals.csv')
reg_stats = pd.read_csv('MRegularSeasonDetailedResults.csv') 

def get_seed(seed_str):
    return int(seed_str[1:3])
seeds['Seed'] = seeds['Seed'].apply(get_seed)

kenpom = rankings[(rankings['SystemName'] == 'POM') & (rankings['RankingDayNum'] == 133)]
kenpom = kenpom[['Season', 'TeamID', 'OrdinalRank']].rename(columns={'OrdinalRank': 'Rank'})

# Calculate estimated possessions
w_poss = reg_stats['WFGA'] - reg_stats['WOR'] + reg_stats['WTO'] + 0.475 * reg_stats['WFTA']
l_poss = reg_stats['LFGA'] - reg_stats['LOR'] + reg_stats['LTO'] + 0.475 * reg_stats['LFTA']

w_data = pd.DataFrame({
    'Season': reg_stats['Season'], 'TeamID': reg_stats['WTeamID'], 
    'Pts': reg_stats['WScore'], 'PtsAll': reg_stats['LScore'], 'Poss': w_poss,
    'FGM': reg_stats['WFGM'], 'FGA': reg_stats['WFGA'], 'FGM3': reg_stats['WFGM3'], 
    'FGA3': reg_stats['WFGA3'], 'FTA': reg_stats['WFTA'], 'TO': reg_stats['WTO'],
    'OR': reg_stats['WOR'], 'Opp_DR': reg_stats['LDR']
})
l_data = pd.DataFrame({
    'Season': reg_stats['Season'], 'TeamID': reg_stats['LTeamID'], 
    'Pts': reg_stats['LScore'], 'PtsAll': reg_stats['WScore'], 'Poss': l_poss,
    'FGM': reg_stats['LFGM'], 'FGA': reg_stats['LFGA'], 'FGM3': reg_stats['LFGM3'], 
    'FGA3': reg_stats['LFGA3'], 'FTA': reg_stats['LFTA'], 'TO': reg_stats['LTO'],
    'OR': reg_stats['LOR'], 'Opp_DR': reg_stats['WDR']
})

# Aggregate total stats by Team and Season
season_totals = pd.concat([w_data, l_data]).groupby(['Season', 'TeamID']).sum().reset_index()

# Calculate the metrics
season_totals['OffRtg'] = 100 * (season_totals['Pts'] / season_totals['Poss'])
season_totals['DefRtg'] = 100 * (season_totals['PtsAll'] / season_totals['Poss'])
season_totals['NetRtg'] = season_totals['OffRtg'] - season_totals['DefRtg']
season_totals['eFG'] = (season_totals['FGM'] + 0.5 * season_totals['FGM3']) / season_totals['FGA']
season_totals['TOV'] = season_totals['TO'] / season_totals['Poss']
season_totals['ORB'] = season_totals['OR'] / (season_totals['OR'] + season_totals['Opp_DR'])
season_totals['FTR'] = season_totals['FTA'] / season_totals['FGA']
season_totals['3PAr'] = season_totals['FGA3'] / season_totals['FGA']

# Calculate 3-Point Percentage
season_totals['3P%'] = season_totals['FGM3'] / season_totals['FGA3']
season_totals['3P%'] = season_totals['3P%'].fillna(0)

# Calculate Native SOS
games = pd.concat([
    reg_stats[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID', 'LTeamID': 'OppID'}),
    reg_stats[['Season', 'LTeamID', 'WTeamID']].rename(columns={'LTeamID': 'TeamID', 'WTeamID': 'OppID'})
])
games = pd.merge(games, season_totals[['Season', 'TeamID', 'NetRtg']].rename(columns={'TeamID': 'OppID', 'NetRtg': 'OppNetRtg'}), on=['Season', 'OppID'], how='left')
sos_df = games.groupby(['Season', 'TeamID'])['OppNetRtg'].mean().reset_index().rename(columns={'OppNetRtg': 'SOS'})
season_totals = pd.merge(season_totals, sos_df, on=['Season', 'TeamID'], how='left')

# Merges for Winning Team
results = pd.merge(results, seeds.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}), on=['Season', 'WTeamID'], how='left')
results = pd.merge(results, kenpom.rename(columns={'TeamID': 'WTeamID', 'Rank': 'WRank'}), on=['Season', 'WTeamID'], how='left')
results = pd.merge(results, season_totals[['Season', 'TeamID', 'NetRtg', 'eFG', 'TOV', 'ORB', 'FTR', '3PAr', '3P%', 'SOS']].rename(columns={'TeamID': 'WTeamID', 'NetRtg': 'WNetRtg', 'eFG': 'WeFG', 'TOV': 'WTOV', 'ORB': 'WORB', 'FTR': 'WFTR', '3PAr': 'W3PAr', '3P%': 'W3P%', 'SOS': 'WSOS'}), on=['Season', 'WTeamID'], how='left')

# Merges for Losing Team
results = pd.merge(results, seeds.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}), on=['Season', 'LTeamID'], how='left')
results = pd.merge(results, kenpom.rename(columns={'TeamID': 'LTeamID', 'Rank': 'LRank'}), on=['Season', 'LTeamID'], how='left')
results = pd.merge(results, season_totals[['Season', 'TeamID', 'NetRtg', 'eFG', 'TOV', 'ORB', 'FTR', '3PAr', '3P%', 'SOS']].rename(columns={'TeamID': 'LTeamID', 'NetRtg': 'LNetRtg', 'eFG': 'LeFG', 'TOV': 'LTOV', 'ORB': 'LORB', 'FTR': 'LFTR', '3PAr': 'L3PAr', '3P%': 'L3P%', 'SOS': 'LSOS'}), on=['Season', 'LTeamID'], how='left')

results = results.dropna()

# Create symmetrical matchups
df_a_wins = results.copy()
df_a_wins['TeamA'] = df_a_wins['WTeamID']
df_a_wins['TeamB'] = df_a_wins['LTeamID']
df_a_wins['SeedDiff'] = df_a_wins['WSeed'] - df_a_wins['LSeed']
df_a_wins['RankDiff'] = df_a_wins['WRank'] - df_a_wins['LRank']
df_a_wins['NetRtgDiff'] = df_a_wins['WNetRtg'] - df_a_wins['LNetRtg'] 
df_a_wins['eFGDiff'] = df_a_wins['WeFG'] - df_a_wins['LeFG'] 
df_a_wins['TOVDiff'] = df_a_wins['WTOV'] - df_a_wins['LTOV'] 
df_a_wins['ORBDiff'] = df_a_wins['WORB'] - df_a_wins['LORB'] 
df_a_wins['FTRDiff'] = df_a_wins['WFTR'] - df_a_wins['LFTR'] 
df_a_wins['3PArDiff'] = df_a_wins['W3PAr'] - df_a_wins['L3PAr'] 
df_a_wins['3P%Diff'] = df_a_wins['W3P%'] - df_a_wins['L3P%'] 
df_a_wins['SOSDiff'] = df_a_wins['WSOS'] - df_a_wins['LSOS'] 
df_a_wins['TeamAWon'] = 1 

df_b_wins = results.copy()
df_b_wins['TeamA'] = df_b_wins['LTeamID']
df_b_wins['TeamB'] = df_b_wins['WTeamID']
df_b_wins['SeedDiff'] = df_b_wins['LSeed'] - df_b_wins['WSeed']
df_b_wins['RankDiff'] = df_b_wins['LRank'] - df_b_wins['WRank']
df_b_wins['NetRtgDiff'] = df_b_wins['LNetRtg'] - df_b_wins['WNetRtg'] 
df_b_wins['eFGDiff'] = df_b_wins['LeFG'] - df_b_wins['WeFG'] 
df_b_wins['TOVDiff'] = df_b_wins['LTOV'] - df_b_wins['WTOV'] 
df_b_wins['ORBDiff'] = df_b_wins['LORB'] - df_b_wins['WORB'] 
df_b_wins['FTRDiff'] = df_b_wins['LFTR'] - df_b_wins['WFTR'] 
df_b_wins['3PArDiff'] = df_b_wins['L3PAr'] - df_b_wins['W3PAr'] 
df_b_wins['3P%Diff'] = df_b_wins['L3P%'] - df_b_wins['W3P%'] 
df_b_wins['SOSDiff'] = df_b_wins['LSOS'] - df_b_wins['WSOS'] # THE MISSING FIX IS HERE!
df_b_wins['TeamAWon'] = 0

model_data = pd.concat([df_a_wins, df_b_wins]).sample(frac=1).reset_index(drop=True)

# ==========================================
# PHASE 2: TRAIN THE XGBOOST MODEL
# ==========================================

latest_season = model_data['Season'].max()
train_data = model_data[model_data['Season'] < latest_season]
test_data = model_data[model_data['Season'] == latest_season]

# The fully upgraded 10-Feature model
features = ['SeedDiff', 'RankDiff', 'NetRtgDiff', 'eFGDiff', 'TOVDiff', 'ORBDiff', 'FTRDiff', '3PArDiff', '3P%Diff', 'SOSDiff']
X_train = train_data[features]
y_train = train_data['TeamAWon']
X_test = test_data[features]
y_test = test_data['TeamAWon']

xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=4, 
    random_state=42,
    eval_metric='logloss' 
)
xgb_model.fit(X_train, y_train)

print(f"\nModel Accuracy (Tested blindly on {latest_season}): {accuracy_score(y_test, xgb_model.predict(X_test)):.2f}\n")

# ==========================================
# PHASE 3: 2026 MATCHUPS & PURE KAGGLE DATA PULL
# ==========================================
print("Loading Latest Live Data directly from Kaggle files...")

try:
    teams_df = pd.read_csv('MTeams.csv')
except FileNotFoundError:
    print("Could not find 'MTeams.csv'. Please make sure it's in your folder!")
    exit()

latest_stat_season = season_totals['Season'].max()
stats_latest = season_totals[season_totals['Season'] == latest_stat_season].set_index('TeamID')

latest_rank_season = rankings['Season'].max()
pom_ranks = rankings[(rankings['SystemName'] == 'POM') & (rankings['Season'] == latest_rank_season)]
ranks_latest = pom_ranks.loc[pom_ranks.groupby('TeamID')['RankingDayNum'].idxmax()].set_index('TeamID')

kaggle_name_translator = {
    'Ohio State': 'Ohio St', 'CA Baptist': 'Cal Baptist', "St. John's": "St John's",
    'Michigan St': 'Michigan St', 'N Dakota St': 'N Dakota St', 'UConn': 'Connecticut',
    'Long Island': 'LIU Brooklyn', 'Utah State': 'Utah St', "Hawai'i": 'Hawaii', 
    'NC State': 'NC State', 'Kennesaw St': 'Kennesaw', 'Miami': 'Miami FL', 
    'Queens': 'Queens NC', 'McNeese': 'McNeese St', "Saint Mary's": "St Mary's CA",
    'Saint Louis': 'St Louis', 'Iowa State': 'Iowa St', 'Tennessee St': 'Tennessee St',
    'Wright St': 'Wright St'
}

matchups_east = [
    {'Region': 'East', 'TeamA': 'Duke', 'SeedA': 1, 'TeamB': 'Siena', 'SeedB': 16},
    {'Region': 'East', 'TeamA': 'Ohio State', 'SeedA': 8, 'TeamB': 'TCU', 'SeedB': 9},
    {'Region': 'East', 'TeamA': "St. John's", 'SeedA': 5, 'TeamB': 'Northern Iowa', 'SeedB': 12},
    {'Region': 'East', 'TeamA': 'Kansas', 'SeedA': 4, 'TeamB': 'CA Baptist', 'SeedB': 13},
    {'Region': 'East', 'TeamA': 'Louisville', 'SeedA': 6, 'TeamB': 'South Florida', 'SeedB': 11},
    {'Region': 'East', 'TeamA': 'Michigan St', 'SeedA': 3, 'TeamB': 'N Dakota St', 'SeedB': 14},
    {'Region': 'East', 'TeamA': 'UCLA', 'SeedA': 7, 'TeamB': 'UCF', 'SeedB': 10},
    {'Region': 'East', 'TeamA': 'UConn', 'SeedA': 2, 'TeamB': 'Furman', 'SeedB': 15}
]

matchups_west = [
    {'Region': 'West', 'TeamA': 'Arizona', 'SeedA': 1, 'TeamB': 'Long Island', 'SeedB': 16},
    {'Region': 'West', 'TeamA': 'Villanova', 'SeedA': 8, 'TeamB': 'Utah State', 'SeedB': 9},
    {'Region': 'West', 'TeamA': 'Wisconsin', 'SeedA': 5, 'TeamB': 'High Point', 'SeedB': 12},
    {'Region': 'West', 'TeamA': 'Arkansas', 'SeedA': 4, 'TeamB': "Hawai'i", 'SeedB': 13},
    {'Region': 'West', 'TeamA': 'BYU', 'SeedA': 6, 'TeamB': 'NC State', 'SeedB': 11},
    {'Region': 'West', 'TeamA': 'Gonzaga', 'SeedA': 3, 'TeamB': 'Kennesaw St', 'SeedB': 14},
    {'Region': 'West', 'TeamA': 'Miami', 'SeedA': 7, 'TeamB': 'Missouri', 'SeedB': 10},
    {'Region': 'West', 'TeamA': 'Purdue', 'SeedA': 2, 'TeamB': 'Queens', 'SeedB': 15}
]

matchups_south = [
    {'Region': 'South', 'TeamA': 'Florida', 'SeedA': 1, 'TeamB': 'Lehigh', 'SeedB': 16},
    {'Region': 'South', 'TeamA': 'Clemson', 'SeedA': 8, 'TeamB': 'Iowa', 'SeedB': 9},
    {'Region': 'South', 'TeamA': 'Vanderbilt', 'SeedA': 5, 'TeamB': 'McNeese', 'SeedB': 12},
    {'Region': 'South', 'TeamA': 'Nebraska', 'SeedA': 4, 'TeamB': 'Troy', 'SeedB': 13},
    {'Region': 'South', 'TeamA': 'North Carolina', 'SeedA': 6, 'TeamB': 'VCU', 'SeedB': 11},
    {'Region': 'South', 'TeamA': 'Illinois', 'SeedA': 3, 'TeamB': 'Penn', 'SeedB': 14},
    {'Region': 'South', 'TeamA': "Saint Mary's", 'SeedA': 7, 'TeamB': 'Texas A&M', 'SeedB': 10},
    {'Region': 'South', 'TeamA': 'Houston', 'SeedA': 2, 'TeamB': 'Idaho', 'SeedB': 15}
]

matchups_midwest = [
    {'Region': 'Midwest', 'TeamA': 'Michigan', 'SeedA': 1, 'TeamB': 'UMBC', 'SeedB': 16},
    {'Region': 'Midwest', 'TeamA': 'Georgia', 'SeedA': 8, 'TeamB': 'Saint Louis', 'SeedB': 9},
    {'Region': 'Midwest', 'TeamA': 'Texas Tech', 'SeedA': 5, 'TeamB': 'Akron', 'SeedB': 12},
    {'Region': 'Midwest', 'TeamA': 'Alabama', 'SeedA': 4, 'TeamB': 'Hofstra', 'SeedB': 13},
    {'Region': 'Midwest', 'TeamA': 'Tennessee', 'SeedA': 6, 'TeamB': 'SMU', 'SeedB': 11},
    {'Region': 'Midwest', 'TeamA': 'Virginia', 'SeedA': 3, 'TeamB': 'Wright St', 'SeedB': 14},
    {'Region': 'Midwest', 'TeamA': 'Kentucky', 'SeedA': 7, 'TeamB': 'Santa Clara', 'SeedB': 10},
    {'Region': 'Midwest', 'TeamA': 'Iowa State', 'SeedA': 2, 'TeamB': 'Tennessee St', 'SeedB': 15}
]

# Fixed to match the real 2026 bracket topology:
# Left Side (East vs South) + Right Side (Midwest vs West)
all_matchups = matchups_east + matchups_south + matchups_midwest + matchups_west

for match in all_matchups:
    match.update({'RankA': 100, 'NetA': 0, 'SOSA': 0, 'eFGA': 0.5, 'TOVA': 0.18, 'ORBA': 0.25, 'FTRA': 0.3, 'P3ArA': 0.35, 'P3PctA': 0.33,
                  'RankB': 100, 'NetB': 0, 'SOSB': 0, 'eFGB': 0.5, 'TOVB': 0.18, 'ORBB': 0.25, 'FTRB': 0.3, 'P3ArB': 0.35, 'P3PctB': 0.33})

    # --- PULL TEAM A ---
    name_a = kaggle_name_translator.get(match['TeamA'], match['TeamA'])
    team_a_rows = teams_df[teams_df['TeamName'] == name_a]['TeamID']
    if len(team_a_rows) > 0:
        tid_a = team_a_rows.values[0]
        if tid_a in stats_latest.index:
            row = stats_latest.loc[tid_a]
            match.update({'NetA': row['NetRtg'], 'eFGA': row['eFG'], 'TOVA': row['TOV'], 
                          'ORBA': row['ORB'], 'FTRA': row['FTR'], 'P3ArA': row['3PAr'], 'P3PctA': row['3P%'], 'SOSA': row['SOS']})
        if tid_a in ranks_latest.index:
            match['RankA'] = ranks_latest.loc[tid_a, 'OrdinalRank']
    else:
        print(f"Missing Kaggle Team A: {match['TeamA']} (Looked for {name_a})")

    # --- PULL TEAM B ---
    name_b = kaggle_name_translator.get(match['TeamB'], match['TeamB'])
    team_b_rows = teams_df[teams_df['TeamName'] == name_b]['TeamID']
    if len(team_b_rows) > 0:
        tid_b = team_b_rows.values[0]
        if tid_b in stats_latest.index:
            row = stats_latest.loc[tid_b]
            match.update({'NetB': row['NetRtg'], 'eFGB': row['eFG'], 'TOVB': row['TOV'], 
                          'ORBB': row['ORB'], 'FTRB': row['FTR'], 'P3ArB': row['3PAr'], 'P3PctB': row['3P%'], 'SOSB': row['SOS']})
        if tid_b in ranks_latest.index:
            match['RankB'] = ranks_latest.loc[tid_b, 'OrdinalRank']
    else:
        print(f"Missing Kaggle Team B: {match['TeamB']} (Looked for {name_b})")

print("Live Kaggle stats perfectly mapped to bracket!")

# ==========================================
# PHASE 4: 100,000 SIMULATIONS & HTML GENERATOR
# ==========================================

print("\n=======================================================")
print("--- RUNNING 100,000 SIMULATIONS & BUILDING HTML BRACKET ---")
print("=======================================================\n")
print("Simulating...")

num_simulations = 100000

team_stats = {}
for match in all_matchups:
    team_stats[match['TeamA']] = {'Team': match['TeamA'], 'Seed': match['SeedA'], 'Rank': match['RankA'], 'NetRtg': match['NetA'], 'eFG': match['eFGA'], 'TOV': match['TOVA'], 'ORB': match['ORBA'], 'FTR': match['FTRA'], '3PAr': match['P3ArA'], '3P%': match['P3PctA'], 'SOS': match['SOSA']}
    team_stats[match['TeamB']] = {'Team': match['TeamB'], 'Seed': match['SeedB'], 'Rank': match['RankB'], 'NetRtg': match['NetB'], 'eFG': match['eFGB'], 'TOV': match['TOVB'], 'ORB': match['ORBB'], 'FTR': match['FTRB'], '3PAr': match['P3ArB'], '3P%': match['P3PctB'], 'SOS': match['SOSB']}

prob_cache = {}

def get_win_prob(team_a, team_b):
    matchup_key = (team_a['Team'], team_b['Team'])
    if matchup_key in prob_cache: return prob_cache[matchup_key]
    
    features_df = pd.DataFrame({
        'SeedDiff': [team_a['Seed'] - team_b['Seed']],
        'RankDiff': [team_a['Rank'] - team_b['Rank']],
        'NetRtgDiff': [team_a['NetRtg'] - team_b['NetRtg']],
        'eFGDiff': [team_a['eFG'] - team_b['eFG']],
        'TOVDiff': [team_a['TOV'] - team_b['TOV']],
        'ORBDiff': [team_a['ORB'] - team_b['ORB']],
        'FTRDiff': [team_a['FTR'] - team_b['FTR']],
        '3PArDiff': [team_a['3PAr'] - team_b['3PAr']],
        '3P%Diff': [team_a['3P%'] - team_b['3P%']],
        'SOSDiff': [team_a['SOS'] - team_b['SOS']] 
    })
    
    prob = xgb_model.predict_proba(features_df)[0][1]
    prob_cache[matchup_key] = prob
    return prob

slot_trackers = {r: [Counter() for _ in range(32 // (2**r))] for r in range(6)}

for sim in range(num_simulations):
    current_round_teams = []
    
    for i, match in enumerate(all_matchups):
        team_a, team_b = team_stats[match['TeamA']], team_stats[match['TeamB']]
        winner = team_a if np.random.rand() < get_win_prob(team_a, team_b) else team_b
        current_round_teams.append(winner)
        slot_trackers[0][i][winner['Team']] += 1
            
    for r in range(1, 6):
        next_round_teams = []
        for i in range(0, len(current_round_teams), 2):
            team1, team2 = current_round_teams[i], current_round_teams[i+1]
            winner = team1 if np.random.rand() < get_win_prob(team1, team2) else team2
            next_round_teams.append(winner)
            slot_trackers[r][i//2][winner['Team']] += 1
        current_round_teams = next_round_teams

bracket_data = {}
round_names = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship Game", "National Champion"]

for r in range(6):
    bracket_data[round_names[r]] = []
    for slot in slot_trackers[r]:
        top_10 = slot.most_common(10)
        top_10_pct = [{"team": t, "prob": f"{(c/num_simulations)*100:.1f}%"} for t, c in top_10]
        bracket_data[round_names[r]].append(top_10_pct)

# --- DEBUG: PRINT THE TOP CONTENDERS ---
print("\n--- Debug Stats (1- and 2-Seeds) ---")
teams_to_check = ['Duke', 'UConn', 'Arizona', 'Purdue', 'Florida', 'Houston', 'Michigan', 'Iowa State']

for t in teams_to_check:
    if t in team_stats:
        stats = team_stats[t]
        print(f"{t.ljust(10)} | Rank: {str(stats['Rank']).ljust(2)} | Net: {stats['NetRtg']:5.1f} | SOS: {stats['SOS']:5.1f} | eFG%: {stats['eFG']:.3f} | TOV%: {stats['TOV']:.3f} | ORB%: {stats['ORB']:.3f} | FTR: {stats['FTR']:.3f} | 3PAr: {stats['3PAr']:.3f} | 3P%: {stats['3P%']:.3f}")
print("--------------------------------------------------------------\n")

# ==========================================
# HTML GENERATOR
# ==========================================

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>2026 March Madness Model Predictions</title>
    <style>
        :root {{
            --bg-0: #0a0e1a;
            --bg-1: #111827;
            --bg-2: #1a2440;
            --border: rgba(255,255,255,0.08);
            --border-hover: #4da3ff;
            --text-0: #e8ecf5;
            --text-1: #9aa5c4;
            --accent: #4da3ff;
            --accent-2: #7c5cff;
            --gold: #f5c542;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: radial-gradient(circle at 20% -10%, #16213d 0%, var(--bg-0) 45%) fixed;
            color: var(--text-0);
            text-align: center;
            margin: 0;
            padding: 40px 20px 60px;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin: 0 0 8px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }}
        body > p {{
            color: var(--text-1);
            font-size: 14px;
            max-width: 620px;
            margin: 0 auto 10px;
            line-height: 1.5;
        }}
        .bracket-wrapper {{
            position: relative;
            margin-top: 36px;
            padding: 24px 0 32px;
            overflow-x: auto;
        }}
        .bracket-container {{
            display: flex;
            justify-content: space-around;
            align-items: stretch;
            position: relative;
            z-index: 2;
            min-width: 1200px;
            gap: 4px;
        }}
        .round-column {{
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            flex: 1;
            margin: 0 12px;
        }}
        .round-header {{
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 22px;
            padding-bottom: 10px;
            color: var(--text-1);
            border-bottom: 2px solid var(--border);
            background: transparent;
        }}
        .round-column:last-child .round-header {{
            color: var(--gold);
            border-bottom-color: var(--gold);
        }}
        .matchup-slot {{
            background: linear-gradient(180deg, var(--bg-2), var(--bg-1));
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 10px;
            margin: 7px 0;
            cursor: default;
            position: relative;
            box-shadow: 0 2px 6px rgba(0,0,0,0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
            min-height: 25px;
            z-index: 5;
        }}
        .matchup-slot:hover {{
            transform: translateY(-2px);
            border-color: var(--border-hover);
            box-shadow: 0 8px 20px rgba(77,163,255,0.25);
        }}
        .round-column:last-child .matchup-slot {{
            border-color: rgba(245,197,66,0.35);
        }}
        .round-column:last-child .matchup-slot:hover {{
            border-color: var(--gold);
            box-shadow: 0 8px 20px rgba(245,197,66,0.3);
        }}
        #bracket-lines {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; }}
        .tooltip {{
            display: none;
            width: 230px;
            background: #0d1424;
            color: var(--text-0);
            text-align: left;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            position: fixed;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.15s ease;
            font-size: 12.5px;
            font-weight: 400;
            line-height: 1.5;
            box-shadow: 0 12px 30px rgba(0,0,0,0.5);
            pointer-events: none;
        }}
        .tooltip.visible {{ display: block; opacity: 1; }}
        .tooltip-title {{
            text-align: center;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }}
        .tooltip-row {{
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 6px;
            margin: 2px 0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .tooltip-bar {{
            position: absolute;
            left: 0; top: 0; bottom: 0;
            background: linear-gradient(90deg, rgba(77,163,255,0.28), rgba(124,92,255,0.05));
            z-index: 0;
        }}
        .tooltip-rank, .tooltip-team, .prob-number {{ position: relative; z-index: 1; }}
        .tooltip-rank {{ color: var(--text-1); width: 16px; flex-shrink: 0; }}
        .tooltip-team {{ flex: 1; margin: 0 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .prob-number {{ color: #5fe0a5; font-weight: 700; flex-shrink: 0; }}
    </style>
</head>
<body>
    <h1>2026 XGBoost Bracket Projections</h1>
    <p>10-feature XGBoost model &middot; 100,000 simulated brackets. Hover over any bracket slot to see the top 10 teams most likely to reach that spot.</p>
    <div class="bracket-wrapper">
        <svg id="bracket-lines">
            <defs>
                <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#4da3ff" stop-opacity="0.15"/>
                    <stop offset="100%" stop-color="#4da3ff" stop-opacity="0.55"/>
                </linearGradient>
            </defs>
        </svg>
        <div class="bracket-container" id="bracket"></div>
    </div>
    <script>
        const bracketData = {json.dumps(bracket_data)};
        const container = document.getElementById("bracket");

        // A single shared tooltip, attached to <body> and positioned with JS.
        // Keeping it out of .bracket-wrapper avoids it being clipped by that
        // element's forced overflow-y (a side effect of overflow-x: auto).
        const tooltip = document.createElement("div");
        tooltip.className = "tooltip";
        document.body.appendChild(tooltip);

        function positionTooltip(slot) {{
            const rect = slot.getBoundingClientRect();
            const gap = 10;
            const ttRect = tooltip.getBoundingClientRect();

            let top = rect.top - ttRect.height - gap;
            if (top < 8) top = rect.bottom + gap; // not enough room above: flip below

            let left = rect.left + rect.width / 2 - ttRect.width / 2;
            left = Math.max(8, Math.min(left, window.innerWidth - ttRect.width - 8));

            tooltip.style.top = `${{top}}px`;
            tooltip.style.left = `${{left}}px`;
        }}

        function hideTooltip() {{ tooltip.classList.remove("visible"); }}

        Object.keys(bracketData).forEach(roundName => {{
            const col = document.createElement("div");
            col.className = "round-column";

            const header = document.createElement("div");
            header.className = "round-header";
            header.innerText = roundName;
            col.appendChild(header);

            bracketData[roundName].forEach((slotData) => {{
                const slot = document.createElement("div");
                slot.className = "matchup-slot";
                slot.innerText = slotData[0] ? slotData[0].team : "TBD";

                let tooltipHTML = `<div class="tooltip-title">Top Odds for This Slot</div>`;
                slotData.forEach((teamEntry, i) => {{
                    const pct = parseFloat(teamEntry.prob) || 0;
                    tooltipHTML += `
                        <div class="tooltip-row">
                            <div class="tooltip-bar" style="width: ${{Math.min(pct, 100)}}%;"></div>
                            <span class="tooltip-rank">${{i+1}}.</span>
                            <span class="tooltip-team">${{teamEntry.team}}</span>
                            <span class="prob-number">${{teamEntry.prob}}</span>
                        </div>
                    `;
                }});

                slot.addEventListener("mouseenter", () => {{
                    tooltip.innerHTML = tooltipHTML;
                    tooltip.classList.add("visible");
                    positionTooltip(slot);
                }});
                slot.addEventListener("mouseleave", hideTooltip);

                col.appendChild(slot);
            }});
            container.appendChild(col);
        }});

        document.querySelector(".bracket-wrapper").addEventListener("scroll", hideTooltip);
        window.addEventListener("scroll", hideTooltip);

        function drawLines() {{
            const svg = document.getElementById("bracket-lines");
            const defs = svg.querySelector("defs");
            svg.innerHTML = "";
            if (defs) svg.appendChild(defs);

            const columns = document.querySelectorAll('.round-column');
            const svgRect = svg.getBoundingClientRect();

            for (let c = 0; c < columns.length - 1; c++) {{
                const currentSlots = columns[c].querySelectorAll('.matchup-slot');
                const nextSlots = columns[c+1].querySelectorAll('.matchup-slot');

                currentSlots.forEach((startNode, i) => {{
                    const nextIndex = Math.floor(i / 2);
                    const endNode = nextSlots[nextIndex];

                    if (endNode) {{
                        const startRect = startNode.getBoundingClientRect();
                        const endRect = endNode.getBoundingClientRect();

                        const startX = startRect.right - svgRect.left;
                        const startY = startRect.top + startRect.height / 2 - svgRect.top;

                        const endX = endRect.left - svgRect.left;
                        const endY = endRect.top + endRect.height / 2 - svgRect.top;

                        const midX = startX + (endX - startX) / 2;

                        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                        path.setAttribute("d", `M ${{startX}} ${{startY}} L ${{midX}} ${{startY}} L ${{midX}} ${{endY}} L ${{endX}} ${{endY}}`);
                        path.setAttribute("fill", "transparent");
                        path.setAttribute("stroke", "url(#lineGradient)");
                        path.setAttribute("stroke-width", "2.5");
                        path.setAttribute("stroke-linejoin", "round");

                        svg.appendChild(path);
                    }}
                }});
            }}
        }}
        setTimeout(drawLines, 100);
        window.addEventListener('resize', drawLines);
    </script>
</body>
</html>
"""

file_path = os.path.join(os.getcwd(), 'interactive_bracket.html')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"SUCCESS!")
print(f"Open this file in your web browser: {file_path}")