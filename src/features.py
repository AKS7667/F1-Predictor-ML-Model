import pandas as pd
import numpy as np

# Load raw data
results = pd.read_parquet("data/f1_all_results.parquet")
laps = pd.read_parquet("data/f1_all_laps.parquet")
weather = pd.read_parquet("data/f1_all_weather.parquet")


# BASE: Race results only

races = results[results.SessionType == 'R'].copy()
races = races[['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status',
               'Season', 'Round', 'Location', 'Circuit']].copy()

races['GridPosition'] = pd.to_numeric(races['GridPosition'], errors='coerce')
races['Position'] = pd.to_numeric(races['Position'], errors='coerce')
races["race_id"] = races["Season"].astype(str) + "_" + races["Round"].astype(str)

# DNFs get position 21
races['Position'] = races['Position'].fillna(21).astype(int)
races['GridPosition'] = races['GridPosition'].fillna(20).astype(int)



# FEATURE 1: Qualifying gap to pole (from results, already there)

quali = results[results.SessionType == 'Q'][['Abbreviation', 'Season', 'Round', 'Q1', 'Q2', 'Q3', 
                                                'TeamName', 'Location', 'Circuit']].copy()
# Convert all times to seconds
for col in ['Q1', 'Q2', 'Q3']:
    quali[col] = pd.to_timedelta(quali[col]).dt.total_seconds()

# Best quali time per driver (furthest session they reached)
quali['QualiBestTime'] = quali[['Q3', 'Q2', 'Q1']].bfill(axis=1).iloc[:, 0]
quali['QualiRank'] = quali.groupby(['Season', 'Round'])['QualiBestTime'].rank(method='first')

# Gap to pole
pole_time = quali.groupby(['Season', 'Round'])['QualiBestTime'].min().reset_index(name='PoleTime')
quali = quali.merge(pole_time, on=['Season', 'Round'], how='left')
quali['QualiGapToPole'] = quali['QualiBestTime'] - quali['PoleTime']

# Only keep columns that are needed
quali = quali[['Abbreviation', 'Season', 'Round', 'QualiGapToPole', 'QualiRank',
               'TeamName', 'Location', 'Circuit']].copy()

races = quali.merge(races, on=['Abbreviation', 'Season', 'Round'],
                    how='outer', suffixes=('_q', ''))

# drop rows that are quali-only in an already-completed race (DNS)
race_happened = races.groupby(['Season', 'Round'])['Position'].transform(lambda x: x.notna().any())
races = races[~(race_happened & races['Position'].isna())]

# for not-yet-run races the results side is NaN — fill from the quali side
for c in ['TeamName', 'Location', 'Circuit']:
    races[c] = races[c].fillna(races[c + '_q'])
    
# for not-yet-run races GridPosition is NaN (no results row) — fill from quali order
races['GridPosition'] = races['GridPosition'].fillna(races['QualiRank'])

# Drop redundant columns
races = races.drop(columns=['QualiRank', 'TeamName_q', 'Location_q', 'Circuit_q'])
races["race_id"] = races["Season"].astype(str) + "_" + races["Round"].astype(str)


# FEATURE 2: Driver form : avg finish last 5 races

races = races.sort_values(['Abbreviation', 'Season', 'Round'])
races['DriverForm'] = (races.groupby('Abbreviation')['Position']
                       .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))


# FEATURE 3: Constructor form : avg finish last 5 races

team_avg = (races.sort_values(['TeamName', 'Season', 'Round'])
            .groupby(['TeamName', 'Season', 'Round'])['Position']
            .mean()
            .reset_index(name='TeamRaceAvg'))

team_avg['ConstructorForm'] = (team_avg.groupby('TeamName')['TeamRaceAvg']
                               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

races = races.merge(team_avg[['TeamName', 'Season', 'Round', 'ConstructorForm']],
                    on=['TeamName', 'Season', 'Round'], how='left')


# FEATURE 4: Circuit overtaking difficulty

# Historical median |finish - grid pos| at each circuit

races['PosChange'] = abs(races['GridPosition'] - races['Position'])

circuit_overtaking = (races.groupby('Circuit')['PosChange']
                      .median()
                      .reset_index(name='CircuitOvertaking'))

races = races.merge(circuit_overtaking, on='Circuit', how='left')


# FEATURE 5: Driver's history at this circuit

races['DriverCircuitAvg'] = (races.sort_values(['Abbreviation', 'Circuit', 'Season'])
                             .groupby(['Abbreviation', 'Circuit'])['Position']
                             .transform(lambda x: x.shift(1).expanding().mean()))


# FEATURE 6: Rain at race start (binary)

race_weather = (weather[weather.SessionType == 'R']
    .sort_values('Time')
    .groupby(['Season', 'Round'])
    .first()
    .reset_index()
    [['Season', 'Round', 'Rainfall']]
)
race_weather['RainAtStart'] = race_weather['Rainfall'].astype(int)
races = races.merge(race_weather[['Season', 'Round', 'RainAtStart']],
                    on=['Season', 'Round'], how='left')


# FEATURE 7: Simple FP2 best lap gap (if available, NaN if not) [Example Feature]

# fp2 = laps[laps.SessionType == 'FP2'].copy()
# fp2 = fp2.dropna(subset=['LapTime'])
# fp2 = fp2[fp2['Deleted'] != True]
# fp2['LapSeconds'] = fp2['LapTime'].dt.total_seconds()

# fp2_best = (fp2.groupby(['Season', 'Round', 'Driver'])['LapSeconds']
#             .min().reset_index(name='FP2BestLap'))

# fp2_fastest = (fp2_best.groupby(['Season', 'Round'])['FP2BestLap']
#                .min().reset_index(name='FP2SessionBest'))

# fp2_best = fp2_best.merge(fp2_fastest, on=['Season', 'Round'], how='left')
# fp2_best['FP2GapToFastest'] = fp2_best['FP2BestLap'] - fp2_best['FP2SessionBest']

# fp2_best = fp2_best.rename(columns={'Driver': 'Abbreviation'})

# races = races.merge(fp2_best[['Abbreviation', 'Season', 'Round', 'FP2GapToFastest']],
#                     on=['Abbreviation', 'Season', 'Round'], how='left')


# FEATURE 8: Is this a DNF-prone driver? (historical DNF rate)

races['IsDNF'] = races['Status'].apply(lambda x: 0 if x == 'Finished' or str(x).startswith('+') else 1)
races['DNFRate'] = (races.sort_values(['Abbreviation', 'Season', 'Round'])
                    .groupby('Abbreviation')['IsDNF']
                    .transform(lambda x: x.shift(1).expanding().mean()))


# CLEAN UP
races = races.drop(columns=['Status', 'PosChange', 'IsDNF'])

races.to_parquet("data/f1_features.parquet", index=False)
print(f"Saved {len(races)} rows with columns:")
print(list(races.columns))