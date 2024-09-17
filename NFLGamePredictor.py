import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  

data = pd.read_csv("NFL_Games.csv", index_col=0)

def determine_home(row):
    if row['Home/Away'] == 'Home':
        return row['Teams']
    else:
        return row['Oppon']
def determine_away(row):
    if row['Home/Away'] == 'Away':
        return row['Teams']
    else:
        return row['Oppon']
    
def determine_homeScore(row):
    if row['Home'] == row['Teams']:
        return row['Tm']
    else:
        return row['Opp']
def determine_awayScore(row):
    if row['Away'] == row['Teams']:
        return row['Tm']
    else:
        return row['Opp']
def determine_1stDownsHome(row):
    if row['Home'] == row['Teams']:
        return row['1stDO']
    else:
        return row['1stDD']
def determine_1stDownsAway(row):
    if row['Away'] == row['Teams']:
        return row['1stDO']
    else:
        return row['1stDD']
def determine_TotalYardsHome(row):
    if row['Home'] == row['Teams']:
        return row['TotYdO']
    else:
        return row['TotYdD']
def determine_TotalYardsAway(row):
    if row['Away'] == row['Teams']:
        return row['TotYdO']
    else:
        return row['TotYdD']
def determine_PassYdsHome(row):
    if row['Home'] == row['Teams']:
        return row['PassYO']
    else:
        return row['PassYD']
def determine_PassYdsAway(row):
    if row['Away'] == row['Teams']:
        return row['PassYO']
    else:
        return row['PassYD']
def determine_RushYdsHome(row):
    if row['Home'] == row['Teams']:
        return row['RushYO']
    else:
        return row['RushYD']
def determine_RushYdsAway(row):
    if row['Away'] == row['Teams']:
        return row['RushYO']
    else:
        return row['RushYD']
def determine_TurnoversOffHome(row):
    if row['Home'] == row['Teams']:
        return row['TOO']
    else:
        return row['TOD']
def determine_TurnoversOffAway(row):
    if row['Away'] == row['Teams']:
        return row['TOO']
    else:
        return row['TOD']
def determine_HomeEPAOFF(row):
    if row['Home'] == row['Teams']:
        return row['Offense']
    else:
        return row['Defense'] * -1
def determine_AwayEPAOFF(row):
    if row['Away'] == row['Teams']:
        return row['Offense']
    else:
        return row['Defense'] * -1
def determine_HomeEPADEF(row):
    if row['Home'] == row['Teams']:
        return row['Defense']
    else:
        return row['Offense'] * -1
def determine_AwayEPADEF(row):
    if row['Away'] == row['Teams']:
        return row['Defense']
    else:
        return row['Offense'] * -1
def determine_HomeEPASPE(row):
    if row['Home'] == row['Teams']:
        return row['Sp. Tms']
    else:
        return row['Sp. Tms'] * -1
def determine_AwayEPASPE(row):
    if row['Away'] == row['Teams']:
        return row['Sp. Tms']
    else:
        return row['Sp. Tms'] * -1
    

data['Home'] = data.apply(determine_home, axis=1)
data['Away'] = data.apply(determine_away, axis=1)
data['HomeScore'] = data.apply(determine_homeScore, axis=1)
data['AwayScore'] = data.apply(determine_awayScore, axis=1)
data['Home1stDowns'] = data.apply(determine_1stDownsHome, axis=1)
data['Away1stDowns'] = data.apply(determine_1stDownsAway, axis=1)
data['HomeTotalYards'] = data.apply(determine_TotalYardsHome, axis=1)
data['AwayTotalYards'] = data.apply(determine_TotalYardsAway, axis=1)
data['HomePassYards'] = data.apply(determine_PassYdsHome, axis=1)
data['AwayPassYards'] = data.apply(determine_PassYdsAway, axis=1)
data['HomeRushYards'] = data.apply(determine_RushYdsHome, axis=1)
data['AwayRushYards'] = data.apply(determine_RushYdsAway, axis=1)
data['HomeTurnoversOFF'] = data.apply(determine_TurnoversOffHome, axis=1)
data['AwayTurnoversOFF'] = data.apply(determine_TurnoversOffAway, axis=1)
data['HomeEPAOFF'] = data.apply(determine_HomeEPAOFF, axis=1)
data['AwayEPAOFF'] = data.apply(determine_AwayEPAOFF, axis=1)
data['HomeEPADEF'] = data.apply(determine_HomeEPADEF, axis=1)
data['AwayEPADEF'] = data.apply(determine_AwayEPADEF, axis=1)
data['HomeEPASPE'] = data.apply(determine_HomeEPASPE, axis=1)
data['AwayEPASPE'] = data.apply(determine_AwayEPASPE, axis=1)

team_mapping = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
    'buf': 'BUF',
    'mia': 'MIA',
    'nyj': 'NYJ',
    'nwe': 'NE',
    'rav': 'BAL',
    'cle': 'CLE',
    'pit': 'PIT',
    'cin': 'CIN',
    'htx': 'HOU',
    'jax': 'JAX',
    'clt': 'IND',
    'oti': 'TEN',
    'kan': 'KC',
    'rai': 'LV',
    'den': 'DEN',
    'sdg': 'LAC',
    'dal': 'DAL',
    'nyg': 'NYG',
    'phi': 'PHI',
    'was': 'WAS',
    'det': 'DET',
    'gnb': 'GB',
    'min': 'MIN',
    'chi': 'CHI',
    'tam': 'TB',
    'nor': 'NO',
    'atl': 'ATL',
    'car': 'CAR',
    'sfo': 'SF',
    'ram': 'LAR',
    'sea': 'SEA',
    'crd': 'ARI',


}

data['Away'] = data['Away'].map(team_mapping)
data['Home'] = data['Home'].map(team_mapping)
data['HomeWon'] = (data['HomeScore']) > (data['AwayScore'])


avg_ptsScored_home = data.groupby('Home')['HomeScore'].mean()
avg_ptsScored_away = data.groupby('Away')['AwayScore'].mean()

avg_ptsAllowed_home = data.groupby('Home')['AwayScore'].mean()
avg_ptsAllowed_away = data.groupby('Away')['HomeScore'].mean()

overall_avg_points_scored = (avg_ptsScored_home + avg_ptsScored_away) / 2
overall_avg_points_allowed = (avg_ptsAllowed_home + avg_ptsAllowed_away) / 2

home_wins = data.groupby('Home')['HomeWon'].sum()
away_wins = data.groupby('Away').apply(lambda x: len(x) - x['HomeWon'].sum(), include_groups=False)

total_games_home = data['Home'].value_counts()
total_games_visitor = data['Away'].value_counts()

overall_wins = home_wins + away_wins
total_games = total_games_home + total_games_visitor

win_rate = overall_wins / total_games

avg_1stDownsHome = data.groupby('Home')['Home1stDowns'].mean()
avg_1stDownsAway = data.groupby('Away')['Away1stDowns'].mean()
overall_avg_1stDowns = (avg_1stDownsAway + avg_1stDownsHome) / 2

avg_PassYdsHome = data.groupby('Home')['HomePassYards'].mean()
avg_PassYdsAway = data.groupby('Away')['AwayPassYards'].mean()
overall_avg_PassYards = (avg_PassYdsAway + avg_PassYdsHome) / 2

avg_TotalYardsHome = data.groupby('Home')['HomeTotalYards'].mean()
avg_TotalYardsAway = data.groupby('Away')['AwayTotalYards'].mean()
overall_avg_TotalYards = (avg_TotalYardsHome + avg_TotalYardsAway) / 2

avg_RushYardsHome = data.groupby('Home')['HomeRushYards'].mean()
avg_RushYardsAway = data.groupby('Away')['AwayRushYards'].mean()
overall_avg_RushYards = (avg_RushYardsHome + avg_RushYardsAway) / 2

avg_TurnoversHomeOFF = data.groupby('Home')['HomeTurnoversOFF'].mean()
avg_TurnoversAwayOFF = data.groupby('Away')['AwayTurnoversOFF'].mean()
overall_avg_Turnovers = (avg_TurnoversHomeOFF + avg_TurnoversAwayOFF) / 2

avg_ptsAllowed_home = data.groupby('Home')['AwayScore'].mean()
avg_ptsAllowed_away = data.groupby('Away')['HomeScore'].mean()
overall_avg_points_scored = (avg_ptsScored_home + avg_ptsScored_away) / 2

avg_EPAHOMEOFF = data.groupby('Home')['HomeEPAOFF'].mean()
avg_EPAAWAYOFF = data.groupby('Away')['AwayEPAOFF'].mean()
overall_avg_EPAOFF = (avg_EPAHOMEOFF + avg_EPAAWAYOFF) / 2

avg_EPAHOMEDEF = data.groupby('Home')['HomeEPADEF'].mean()
avg_EPAAWAYDEF = data.groupby('Away')['AwayEPADEF'].mean()
overall_avg_EPADEF = (avg_EPAHOMEDEF + avg_EPAAWAYDEF) / 2

avg_EPAHOMESPE = data.groupby('Home')['HomeEPASPE'].mean()
avg_EPAAWAYSPE = data.groupby('Away')['AwayEPASPE'].mean()
overall_avg_EPASPE = (avg_EPAHOMESPE + avg_EPAAWAYSPE) / 2

team_features = pd.DataFrame({
    'AvgPointsScored': overall_avg_points_scored,
    'AvgPointsAllowed': overall_avg_points_allowed,
    'WinRate': win_rate,
    'Avg1stDowns': overall_avg_1stDowns,
    'AvgTotalYards': overall_avg_TotalYards,
    'AvgPassYards': overall_avg_PassYards,
    'AvgRushYards': overall_avg_RushYards,
    'AvgTurnoversOFF': overall_avg_Turnovers,
    'AvgEPAOFF': overall_avg_EPAOFF,
    'AvgEPADEF': overall_avg_EPADEF,
    'AvgEPASPE': overall_avg_EPASPE,
})

team_features.reset_index(inplace=True)
team_features.rename(columns={'index': 'Team'}, inplace=True)

upcoming_games = pd.read_csv("FutureGames2.csv", index_col=0)
upcoming_encoded_home = upcoming_games.merge(team_features, left_on='Home', right_on='Team', how='left')
upcoming_encoded_both = upcoming_encoded_home.merge(team_features, left_on='Away', right_on='Team', suffixes=('_Home', '_Away'), how='left')

for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'Avg1stDowns', 'AvgTotalYards', 'AvgPassYards', 'AvgRushYards', 'AvgTurnoversOFF', 'AvgEPAOFF', 'AvgEPADEF','AvgEPASPE']:
    upcoming_encoded_both[f'Diff_{col}'] = upcoming_encoded_both[f'{col}_Home'] - upcoming_encoded_both[f'{col}_Away']


upcoming_encoded_final = upcoming_encoded_both[['Home', 'Away'] + [col for col in upcoming_encoded_both.columns if 'Diff' in col]]

training_encoded_home = data.merge(team_features, left_on='Home', right_on='Team', how='left')
training_encoded_both = training_encoded_home.merge(team_features, left_on='Away', right_on='Team', suffixes=('_Home', '_Away'), how='left')

for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'Avg1stDowns', 'AvgTotalYards', 'AvgPassYards', 'AvgRushYards', 'AvgTurnoversOFF', 'AvgEPAOFF', 'AvgEPADEF','AvgEPASPE']:
    training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Away']

training_data = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
training_labels = training_encoded_both['HomeWon']

rf = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression (max_iter=1000)
cross_val_scores = cross_val_score(rf, training_data, training_labels, cv=5)
cross_val_scores_mean = cross_val_scores.mean()

X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25)
rf.fit(X_train, y_train)
 
y_pred = rf.predict(X_test)
 
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

upcoming_game_probs = rf.predict_proba(upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff' in col]])

upcoming_encoded_final['HomeWinProbability'] = upcoming_game_probs[:, 1]

upcoming_predictions = upcoming_encoded_final[['Home', 'Away', 'HomeWinProbability']].sort_values(by='HomeWinProbability', ascending=False)

print(upcoming_predictions)