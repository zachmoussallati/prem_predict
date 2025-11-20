import nest_asyncio
nest_asyncio.apply()
import asyncio
from understat import Understat
import aiohttp
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------- ASYNC DATA LOADER ----------
async def fetch_understat_data(years):
    teams, all_seasons_data = set(), []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for year in years:
            print(f"Getting EPL season: {year}")
            try:
                league_data = await understat.get_league_table("EPL", year)
                
                # league_data is a list of lists: first row is headers, rest are data
                if isinstance(league_data, list) and len(league_data) > 0:
                    headers = league_data[0]  # First row contains column names
                    
                    # Process each team row (skip header row)
                    for row in league_data[1:]:
                        if isinstance(row, list) and len(row) == len(headers):
                            # Convert list to dict using headers
                            team_dict = dict(zip(headers, row))
                            team_dict["season"] = year
                            all_seasons_data.append(team_dict)
                            if "Team" in team_dict:
                                teams.add(team_dict["Team"])
            except Exception as e:
                print(f"  Error fetching {year}: {e}")
    return all_seasons_data

# ---------- FETCH PLAYER DATA ----------
async def fetch_player_data(years):
    """
    Fetch player-level data from Understat for each team/season.
    Falls back to synthetic/estimated player features based on team stats.
    """
    player_data = []
    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for year in years:
            print(f"  Getting player data for season: {year}")
            try:
                # Try to fetch player stats by team
                players_by_team = await understat.get_league_players("EPL", year)
                
                if isinstance(players_by_team, dict) and len(players_by_team) > 0:
                    for team_name, players in players_by_team.items():
                        if isinstance(players, list):
                            for player in players:
                                if isinstance(player, dict):
                                    player_info = {
                                        "Team": team_name,
                                        "season": year,
                                        "player_name": player.get("player_name", ""),
                                        "position": player.get("position", ""),
                                        "xG": pd.to_numeric(player.get("xG", 0), errors="coerce") or 0,
                                        "goals": pd.to_numeric(player.get("goals", 0), errors="coerce") or 0,
                                        "minutes": pd.to_numeric(player.get("minutes", 0), errors="coerce") or 0,
                                    }
                                    player_data.append(player_info)
                else:
                    print(f"    No player data available from API for {year}")
            except Exception as e:
                print(f"    Unable to fetch players from API for {year}: {type(e).__name__}")
    
    return player_data

# ---------- SYNTHETIC PLAYER FEATURES (for when API data unavailable) ----------
def generate_synthetic_player_features(team_stats):
    """
    Generate realistic synthetic player-level features based on team aggregate stats.
    This provides a fallback when direct player data is unavailable.
    """
    player_data = []
    
    for idx, row in team_stats.iterrows():
        team = row['Team']
        season = row['season']
        
        # Extract team-level metrics
        team_xg = pd.to_numeric(row.get('xG', 1), errors='coerce') or 1
        team_goals = pd.to_numeric(row.get('G', 1), errors='coerce') or 1
        team_npxg = pd.to_numeric(row.get('NPxG', 1), errors='coerce') or 1
        
        # Estimate striker quality from xG efficiency
        striker_efficiency = team_goals / max(team_xg, 1)
        
        # Generate synthetic player records
        # Top striker (estimates top scorer based on team xG)
        top_xg = team_xg * 0.30  # Top player ~30% of team xG
        top_goals = top_xg * striker_efficiency
        player_data.append({
            'Team': team,
            'season': season,
            'player_name': f'Top Striker {team}',
            'position': 'ST',
            'xG': top_xg,
            'goals': top_goals,
            'minutes': 2500,
        })
        
        # 2nd-4th scorers
        for i in range(2, 5):
            xg_estimate = team_xg * (0.15 - i*0.02) 
            goals_estimate = xg_estimate * striker_efficiency
            player_data.append({
                'Team': team,
                'season': season,
                'player_name': f'Scorer {i} {team}',
                'position': 'ST' if i == 2 else ('CAM' if i == 3 else 'MF'),
                'xG': max(xg_estimate, 0),
                'goals': max(goals_estimate, 0),
                'minutes': 1800 - i*200,
            })
        
        # Midfielders and defenders (contribute less to xG)
        for i in range(5, 12):
            xg_estimate = team_npxg * 0.08 / 7
            player_data.append({
                'Team': team,
                'season': season,
                'player_name': f'Midfielder {i-4} {team}',
                'position': 'MF',
                'xG': xg_estimate,
                'goals': 0,
                'minutes': 1200 - (i-5)*150,
            })
    
    return pd.DataFrame(player_data)

# ---------- ENGINEER TEAM-LEVEL PLAYER FEATURES ----------
def engineer_player_features(player_df):
    """Create team-level features from player data"""
    team_features = {}
    
    for (team, season), group in player_df.groupby(['Team', 'season']):
        key = (team, season)
        
        # Feature 1: Sum of top 5 players' xG
        top_5_xg = group.nlargest(5, 'xG')['xG'].sum()
        
        # Feature 2: Number of players with >10 goals
        players_10plus_goals = (group['goals'] > 10).sum()
        
        # Feature 3: Average xG per player (minimum 90 minutes played)
        qualified_players = group[group['minutes'] >= 90]
        avg_xg_per_player = qualified_players['xG'].mean() if len(qualified_players) > 0 else 0
        
        # Feature 4: Total squad xG
        total_xg = group['xG'].sum()
        
        # Feature 5: Squad size (players with >0 minutes)
        squad_size = (group['minutes'] > 0).sum()
        
        # Feature 6: Attacking players count (strikers/forwards)
        attacking = (group['position'].isin(['ST', 'CF', 'LW', 'RW', 'CAM'])).sum()
        
        team_features[key] = {
            'top_5_xG': top_5_xg,
            'players_10plus_goals': players_10plus_goals,
            'avg_xG_per_qualified_player': avg_xg_per_player,
            'total_squad_xG': total_xg,
            'squad_size': squad_size,
            'attacking_players': attacking,
        }
    
    return team_features

# ---------- ODDS CONVERSION FUNCTIONS ----------
def probabilities_to_odds(probabilities, overround=0.05, format='decimal'):
    """
    Convert win probabilities to market odds with configurable margin.
    
    Args:
        probabilities: list or array of probabilities (0-1)
        overround: market margin as decimal (0.05 = 5% margin)
                   Higher overround = lower odds for players
        format: 'decimal' (1.50, 2.00, etc) or 'fractional' (1/2, even, 3/1, etc)
    
    Returns:
        dict with 'implied_prob', 'decimal_odds', 'fractional_odds', 'payout'
    """
    probs = np.array(probabilities, dtype=float)
    
    # Calculate total probability (usually sums to 1.0)
    total_prob = probs.sum()
    
    # Add overround to each probability
    # Total probabilities will sum to (1 + overround)
    adjusted_probs = probs / (1 + overround)
    
    # Decimal odds = 1 / implied_probability
    decimal_odds = 1.0 / adjusted_probs
    
    # Create results dictionary
    results = []
    for prob, adj_prob, odds in zip(probs, adjusted_probs, decimal_odds):
        # Fractional odds calculation
        frac_odds = odds - 1
        
        # Format fractional odds as readable string
        if frac_odds < 0.01:
            frac_str = f"1/{int(1/frac_odds)}"
        elif abs(frac_odds - round(frac_odds)) < 0.01:
            frac_str = f"{int(round(frac_odds))}/1"
        else:
            # Simplify the fraction
            from fractions import Fraction
            frac = Fraction(odds - 1).limit_denominator(100)
            frac_str = f"{frac.numerator}/{frac.denominator}"
        
        results.append({
            'original_prob': prob,
            'implied_prob': adj_prob,
            'decimal_odds': odds,
            'fractional_odds': frac_str,
            'payout_at_1': odds,  # Return for 1 unit stake
            'overround': overround
        })
    
    return results

def create_odds_table(predictions_df, teams, probabilities, season, overround=0.05):
    """
    Create a formatted odds table from team predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        teams: list of team names
        probabilities: list of win probabilities
        season: season identifier
        overround: market margin (default 5%)
    
    Returns:
        DataFrame with odds information
    """
    odds_data = probabilities_to_odds(probabilities, overround=overround)
    
    table = pd.DataFrame({
        'Team': teams,
        'Original_Prob': [o['original_prob'] for o in odds_data],
        'Implied_Prob': [o['implied_prob'] for o in odds_data],
        'Decimal_Odds': [o['decimal_odds'] for o in odds_data],
        'Fractional_Odds': [o['fractional_odds'] for o in odds_data],
        'Payout_@_1_Unit': [o['payout_at_1'] for o in odds_data],
    })
    
    # Sort by decimal odds (best odds first, worst last)
    table = table.sort_values('Decimal_Odds').reset_index(drop=True)
    table['Rank'] = range(1, len(table) + 1)
    
    return table[['Rank', 'Team', 'Original_Prob', 'Implied_Prob', 
                   'Decimal_Odds', 'Fractional_Odds', 'Payout_@_1_Unit']]

# ---------- MONTE CARLO SIMULATION ----------
def monte_carlo_season_simulation(teams, win_probabilities, num_simulations=10000):
    """
    Simulate the entire season outcome multiple times using Monte Carlo sampling.
    
    For each simulation:
    - Simulate a full round-robin season (each team plays every other team twice)
    - For each match, sample outcome based on team strength (win probabilities)
    - Accumulate points (3 for win, 1 for draw, 0 for loss)
    - Determine league winner based on final points
    - Record the winner
    
    Args:
        teams: list of team names
        win_probabilities: numpy array of each team's league win probability
        num_simulations: number of season simulations to run (default 10,000)
    
    Returns:
        empirical_win_probs: dict mapping team -> empirical win frequency
        winner_counts: dict mapping team -> count of simulations won
    """
    n_teams = len(teams)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    
    # Track wins for each team
    winner_counts = {team: 0 for team in teams}
    
    print(f"\n[MONTE CARLO SIMULATION]")
    print(f"Running {num_simulations:,} season simulations...")
    print(f"Teams: {n_teams}, Win Probs: {win_probabilities.min():.4f} - {win_probabilities.max():.4f}")
    
    for sim_num in range(num_simulations):
        # Show progress every 1000 simulations
        if (sim_num + 1) % 1000 == 0:
            print(f"  Progress: {sim_num + 1:,}/{num_simulations:,} simulations...", end='\r')
        
        # Initialize points for all teams
        points = {team: 0 for team in teams}
        
        # Simulate all matches (round-robin: each team plays every other team twice)
        for home_idx, home_team in enumerate(teams):
            for away_idx, away_team in enumerate(teams):
                if home_idx == away_idx:
                    continue  # Skip self-matches
                
                # Get team strengths from league win probabilities
                home_strength = win_probabilities[home_idx]
                away_strength = win_probabilities[away_idx]
                
                # Normalize to create match-specific probabilities
                # Based on relative strength ratio
                total_strength = home_strength + away_strength
                if total_strength > 0:
                    home_match_prob = home_strength / total_strength
                else:
                    home_match_prob = 0.5
                
                # Assume draw probability (empirical average from real EPL)
                draw_prob = 0.25
                
                # Adjust win probabilities for draw rate
                home_win_prob = home_match_prob * (1 - draw_prob)
                away_win_prob = (1 - home_match_prob) * (1 - draw_prob)
                
                # Sample match outcome
                outcome = np.random.choice(
                    ['home_win', 'draw', 'away_win'],
                    p=[home_win_prob, draw_prob, away_win_prob]
                )
                
                # Assign points based on outcome
                if outcome == 'home_win':
                    points[home_team] += 3
                elif outcome == 'draw':
                    points[home_team] += 1
                    points[away_team] += 1
                else:  # away_win
                    points[away_team] += 3
        
        # Determine winner (team with most points)
        # In case of tie, pick first one (rare event, would use goal difference in real league)
        winner = max(points, key=points.get)
        winner_counts[winner] += 1
    
    print(f"  Progress: {num_simulations:,}/{num_simulations:,} simulations... ✓")
    
    # Calculate empirical probabilities
    empirical_win_probs = {
        team: winner_counts[team] / num_simulations 
        for team in teams
    }
    
    return empirical_win_probs, winner_counts

# Main workflow
years = list(range(2016, 2026))  # Extended: 2016-2025 (10 seasons if available)

print("="*60)
print("FETCHING LEAGUE DATA")
print("="*60)
all_seasons_data = asyncio.run(fetch_understat_data(years))
df = pd.DataFrame(all_seasons_data)
print("Columns in df:", df.columns.tolist())

print("\n" + "="*60)
print("FETCHING PLAYER DATA")
print("="*60)
all_player_data = asyncio.run(fetch_player_data(years))
player_df = pd.DataFrame(all_player_data)

# Check if player data was retrieved
if len(player_df) == 0:
    print("No player data available from API. Proceeding with team-level features only.")
    feature_cols_new = []
else:
    print(f"Total player records: {len(player_df)}")
    print("\nSample player data:")
    print(player_df[['Team', 'season', 'player_name', 'position', 'xG', 'goals']].head(15))

    # Engineer team-level features from player data
    print("\n" + "="*60)
    print("ENGINEERING PLAYER-BASED FEATURES")
    print("="*60)
    team_player_features = engineer_player_features(player_df)

    # Create feature dataframe
    features_list = []
    for (team, season), features in team_player_features.items():
        row = {'Team': team, 'season': str(season), **features}
        features_list.append(row)

    player_features_df = pd.DataFrame(features_list)
    player_features_df['season'] = player_features_df['season'].astype(str)
    print(f"Created player features for {len(player_features_df)} team-seasons")
    print("\nSample player features:")
    print(player_features_df[['Team', 'season', 'top_5_xG', 'players_10plus_goals', 'squad_size']].head(10))

    # Merge player features with main dataframe
    df['season'] = df['season'].astype(str)
    df = df.merge(player_features_df, on=['Team', 'season'], how='left')

    # Fill NaN values for teams without player data
    feature_cols_new = ['top_5_xG', 'players_10plus_goals', 'avg_xG_per_qualified_player', 
                        'total_squad_xG', 'squad_size', 'attacking_players']
    for col in feature_cols_new:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"\nPlayer features successfully merged. New shape: {df.shape}")
    print(f"Available player-based features: {feature_cols_new}")

# Use numeric features for ML; exclude metadata fields
drops = [
    "Team", "id", "history", "season", "position", "notes", "winner"
]

# Winner label logic using correct PTS column
df["PTS"] = pd.to_numeric(df["PTS"])
df["season"] = df["season"].astype(str)
df["winner"] = 0
for s in df["season"].unique():
    idx = df[df["season"]==s]["PTS"].idxmax()
    df.at[idx, "winner"] = 1

# ------- Feature Engineering --------
# Convert necessary columns to numeric
df["G"] = pd.to_numeric(df["G"], errors="coerce")
df["GA"] = pd.to_numeric(df["GA"], errors="coerce")
df["M"] = pd.to_numeric(df["M"], errors="coerce")
df["xG"] = pd.to_numeric(df["xG"], errors="coerce")
df["xGA"] = pd.to_numeric(df["xGA"], errors="coerce")

# (a) Goal difference (G - GA)
df["GD"] = df["G"] - df["GA"]

# (b) Points per match (PTS / M)
df["PPM"] = df["PTS"] / df["M"]

# (c) xG differential (xG - xGA)
df["xGD"] = df["xG"] - df["xGA"]

print("New features created: GD (Goal Difference), PPM (Points Per Match), xGD (xG Differential)")
print(f"Sample values:\n{df[['Team', 'season', 'GD', 'PPM', 'xGD']].head(10)}\n")

# Show player-based features if available
if len(feature_cols_new) > 0:
    print("Player-based features engineered:")
    for col in feature_cols_new:
        if col in df.columns:
            print(f"  - {col}: {df[col].describe().to_dict()}")

# Build feature columns list (now includes new features)
feature_cols = [
    col for col in df.columns if col not in drops and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors='coerce'))
]

print("\n" + "="*60)
print(f"Columns for ML: {feature_cols}")
print(f"Total features: {len(feature_cols)}")
print("="*60)
print(df.head())

# ------- Rolling Window Validation --------
seasons_sorted = sorted(df["season"].unique())
print(f"\nAvailable seasons: {seasons_sorted}")
print(f"Total seasons: {len(seasons_sorted)}\n")

# Store results for each window
window_results = []
all_predictions = []
overfitting_analysis = []

# ---------- MODEL COMPARISON FUNCTION ----------
def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train both XGBoost and MLP models and compare performance.
    
    Returns:
        results_dict: Contains metrics for both models
        xgb_model: Trained XGBoost model
        mlp_model: Trained MLP model
        probas: Tuple of (xgb_proba, mlp_proba) for test set
    """
    results = {}
    
    # ===== XGBOOST MODEL =====
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        verbose=0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # XGBoost predictions
    xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_train_pred = (xgb_train_proba == xgb_train_proba.max()).astype(int)
    xgb_test_pred = (xgb_test_proba == xgb_test_proba.max()).astype(int)
    
    results['xgb'] = {
        'model': xgb_model,
        'train_acc': accuracy_score(y_train, xgb_train_pred),
        'test_acc': accuracy_score(y_test, xgb_test_pred),
        'train_auc': roc_auc_score(y_train, xgb_train_proba),
        'test_auc': roc_auc_score(y_test, xgb_test_proba),
        'test_proba': xgb_test_proba
    }
    results['xgb']['acc_gap'] = results['xgb']['train_acc'] - results['xgb']['test_acc']
    results['xgb']['auc_gap'] = results['xgb']['train_auc'] - results['xgb']['test_auc']
    
    # ===== MLP MODEL =====
    # MLP architecture: Input(20) -> Hidden1(64) -> Hidden2(32) -> Hidden3(16) -> Output(1)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=False,
        verbose=0
    )
    mlp_model.fit(X_train, y_train)
    
    # MLP predictions
    mlp_train_proba = mlp_model.predict_proba(X_train)[:, 1]
    mlp_test_proba = mlp_model.predict_proba(X_test)[:, 1]
    mlp_train_pred = (mlp_train_proba == mlp_train_proba.max()).astype(int)
    mlp_test_pred = (mlp_test_proba == mlp_test_proba.max()).astype(int)
    
    results['mlp'] = {
        'model': mlp_model,
        'train_acc': accuracy_score(y_train, mlp_train_pred),
        'test_acc': accuracy_score(y_test, mlp_test_pred),
        'train_auc': roc_auc_score(y_train, mlp_train_proba),
        'test_auc': roc_auc_score(y_test, mlp_test_proba),
        'test_proba': mlp_test_proba
    }
    results['mlp']['acc_gap'] = results['mlp']['train_acc'] - results['mlp']['test_acc']
    results['mlp']['auc_gap'] = results['mlp']['train_auc'] - results['mlp']['test_auc']
    
    return results, xgb_model, mlp_model

# Model comparison storage
model_comparison_results = []
all_predictions = []
overfitting_analysis = []

# Rolling window: train on 3 consecutive seasons, test on the next 1 season
# Dynamically generate windows from all available seasons
num_windows = max(1, len(seasons_sorted) - 3)  # Ensure at least 1 window if we have 4+ seasons
for window_start in range(num_windows):
    train_seasons = seasons_sorted[window_start:window_start + 3]
    test_season = seasons_sorted[window_start + 3]
    
    print(f"\nWindow {window_start + 1}: Train on {train_seasons}, Test on {test_season}")
    print("-" * 60)
    
    df_train = df[df["season"].isin(train_seasons)].copy()
    df_test = df[df["season"] == test_season].copy()
    
    X_train = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_train = df_train["winner"].values
    X_test = df_test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_test = df_test["winner"].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and compare models
    model_results, xgb_model, mlp_model = train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_cols)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print()
    
    # ===== XGBOOST RESULTS =====
    xgb_res = model_results['xgb']
    print("XGBOOST MODEL:")
    print(f"  Train Accuracy: {xgb_res['train_acc']:.4f}, Test Accuracy: {xgb_res['test_acc']:.4f} (Gap: {xgb_res['acc_gap']:+.4f})")
    print(f"  Train ROC AUC:  {xgb_res['train_auc']:.4f}, Test ROC AUC:  {xgb_res['test_auc']:.4f} (Gap: {xgb_res['auc_gap']:+.4f})")
    
    if xgb_res['acc_gap'] > 0.05 or xgb_res['auc_gap'] > 0.05:
        print("  ⚠️  WARNING: Possible overfitting detected (gap > 5%)")
    elif xgb_res['acc_gap'] < -0.05 or xgb_res['auc_gap'] < -0.05:
        print("  ⚠️  WARNING: Possible underfitting detected (negative gap)")
    else:
        print("  ✅ No significant overfitting detected")
    
    # ===== MLP RESULTS =====
    mlp_res = model_results['mlp']
    print()
    print("MLP NEURAL NETWORK:")
    print(f"  Train Accuracy: {mlp_res['train_acc']:.4f}, Test Accuracy: {mlp_res['test_acc']:.4f} (Gap: {mlp_res['acc_gap']:+.4f})")
    print(f"  Train ROC AUC:  {mlp_res['train_auc']:.4f}, Test ROC AUC:  {mlp_res['test_auc']:.4f} (Gap: {mlp_res['auc_gap']:+.4f})")
    
    if mlp_res['acc_gap'] > 0.05 or mlp_res['auc_gap'] > 0.05:
        print("  ⚠️  WARNING: Possible overfitting detected (gap > 5%)")
    elif mlp_res['acc_gap'] < -0.05 or mlp_res['auc_gap'] < -0.05:
        print("  ⚠️  WARNING: Possible underfitting detected (negative gap)")
    else:
        print("  ✅ No significant overfitting detected")
    
    # ===== MODEL COMPARISON =====
    print()
    print("MODEL COMPARISON:")
    acc_diff = xgb_res['test_acc'] - mlp_res['test_acc']
    auc_diff = xgb_res['test_auc'] - mlp_res['test_auc']
    
    if abs(acc_diff) < 0.001:
        winner_acc = "TIE"
    else:
        winner_acc = "🔷 XGBoost" if acc_diff > 0 else "🔶 MLP"
    
    if abs(auc_diff) < 0.001:
        winner_auc = "TIE"
    else:
        winner_auc = "🔷 XGBoost" if auc_diff > 0 else "🔶 MLP"
    
    print(f"  Accuracy: XGBoost {xgb_res['test_acc']:.4f} vs MLP {mlp_res['test_acc']:.4f} → {winner_acc} (diff: {acc_diff:+.4f})")
    print(f"  ROC AUC:  XGBoost {xgb_res['test_auc']:.4f} vs MLP {mlp_res['test_auc']:.4f} → {winner_auc} (diff: {auc_diff:+.4f})")
    
    # Store comparison results
    model_comparison_results.append({
        'window': window_start + 1,
        'test_season': test_season,
        'xgb_train_acc': xgb_res['train_acc'],
        'xgb_test_acc': xgb_res['test_acc'],
        'xgb_train_auc': xgb_res['train_auc'],
        'xgb_test_auc': xgb_res['test_auc'],
        'xgb_acc_gap': xgb_res['acc_gap'],
        'xgb_auc_gap': xgb_res['auc_gap'],
        'mlp_train_acc': mlp_res['train_acc'],
        'mlp_test_acc': mlp_res['test_acc'],
        'mlp_train_auc': mlp_res['train_auc'],
        'mlp_test_auc': mlp_res['test_auc'],
        'mlp_acc_gap': mlp_res['acc_gap'],
        'mlp_auc_gap': mlp_res['auc_gap']
    })
    
    # ===== OVERFITTING DIAGNOSTICS =====
    overfitting_analysis.append({
        'window': window_start + 1,
        'xgb_train_acc': xgb_res['train_acc'],
        'xgb_test_acc': xgb_res['test_acc'],
        'xgb_train_auc': xgb_res['train_auc'],
        'xgb_test_auc': xgb_res['test_auc'],
        'xgb_acc_gap': xgb_res['acc_gap'],
        'xgb_auc_gap': xgb_res['auc_gap'],
        'mlp_train_acc': mlp_res['train_acc'],
        'mlp_test_acc': mlp_res['test_acc'],
        'mlp_train_auc': mlp_res['train_auc'],
        'mlp_test_auc': mlp_res['test_auc'],
        'mlp_acc_gap': mlp_res['acc_gap'],
        'mlp_auc_gap': mlp_res['auc_gap'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': len(feature_cols)
    })
    
    window_results.append({
        'train_seasons': train_seasons,
        'test_season': test_season,
        'xgb_accuracy': xgb_res['test_acc'],
        'xgb_auc': xgb_res['test_auc'],
        'mlp_accuracy': mlp_res['test_acc'],
        'mlp_auc': mlp_res['test_auc'],
        'xgb_model': xgb_model,
        'mlp_model': mlp_model
    })
    
    # Store predictions with metadata (using XGBoost for simulations)
    df_test_pred = df_test.copy()
    df_test_pred["win_prob"] = xgb_res['test_proba']
    df_test_pred["win_prob_mlp"] = mlp_res['test_proba']
    df_test_pred["window"] = window_start + 1
    all_predictions.append(df_test_pred)

# Print summary statistics
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

# Extract model performance metrics
xgb_accs = [r['xgb_test_acc'] for r in model_comparison_results]
xgb_aucs = [r['xgb_test_auc'] for r in model_comparison_results]
xgb_train_accs = [r['xgb_train_acc'] for r in model_comparison_results]
xgb_train_aucs = [r['xgb_train_auc'] for r in model_comparison_results]

mlp_accs = [r['mlp_test_acc'] for r in model_comparison_results]
mlp_aucs = [r['mlp_test_auc'] for r in model_comparison_results]
mlp_train_accs = [r['mlp_train_acc'] for r in model_comparison_results]
mlp_train_aucs = [r['mlp_train_auc'] for r in model_comparison_results]

print(f"\n{'XGBOOST - TRAINING PERFORMANCE':^60}")
print(f"Average Train Accuracy: {sum(xgb_train_accs)/len(xgb_train_accs):.4f} (+/- {pd.Series(xgb_train_accs).std():.4f})")
print(f"Average Train ROC AUC:  {sum(xgb_train_aucs)/len(xgb_train_aucs):.4f} (+/- {pd.Series(xgb_train_aucs).std():.4f})")

print(f"\n{'XGBOOST - TEST PERFORMANCE':^60}")
print(f"Average Test Accuracy: {sum(xgb_accs)/len(xgb_accs):.4f} (+/- {pd.Series(xgb_accs).std():.4f})")
print(f"Average Test ROC AUC:  {sum(xgb_aucs)/len(xgb_aucs):.4f} (+/- {pd.Series(xgb_aucs).std():.4f})")

print(f"\n{'MLP - TRAINING PERFORMANCE':^60}")
print(f"Average Train Accuracy: {sum(mlp_train_accs)/len(mlp_train_accs):.4f} (+/- {pd.Series(mlp_train_accs).std():.4f})")
print(f"Average Train ROC AUC:  {sum(mlp_train_aucs)/len(mlp_train_aucs):.4f} (+/- {pd.Series(mlp_train_aucs).std():.4f})")

print(f"\n{'MLP - TEST PERFORMANCE':^60}")
print(f"Average Test Accuracy: {sum(mlp_accs)/len(mlp_accs):.4f} (+/- {pd.Series(mlp_accs).std():.4f})")
print(f"Average Test ROC AUC:  {sum(mlp_aucs)/len(mlp_aucs):.4f} (+/- {pd.Series(mlp_aucs).std():.4f})")

# Model comparison
print(f"\n{'WINNER - ACCURACY':^60}")
avg_xgb_acc = sum(xgb_accs)/len(xgb_accs)
avg_mlp_acc = sum(mlp_accs)/len(mlp_accs)
winner = "🔷 XGBoost" if avg_xgb_acc > avg_mlp_acc else ("🔶 MLP" if avg_mlp_acc > avg_xgb_acc else "TIE")
print(f"XGBoost: {avg_xgb_acc:.4f} | MLP: {avg_mlp_acc:.4f} | Diff: {avg_xgb_acc - avg_mlp_acc:+.4f} → {winner}")

print(f"\n{'WINNER - ROC AUC':^60}")
avg_xgb_auc = sum(xgb_aucs)/len(xgb_aucs)
avg_mlp_auc = sum(mlp_aucs)/len(mlp_aucs)
winner = "🔷 XGBoost" if avg_xgb_auc > avg_mlp_auc else ("🔶 MLP" if avg_mlp_auc > avg_xgb_auc else "TIE")
print(f"XGBoost: {avg_xgb_auc:.4f} | MLP: {avg_mlp_auc:.4f} | Diff: {avg_xgb_auc - avg_mlp_auc:+.4f} → {winner}")

# Overfitting Analysis
print("\n" + "="*60)
print("OVERFITTING ANALYSIS - XGBOOST")
print("="*60)

xgb_acc_gaps = [r['xgb_acc_gap'] for r in model_comparison_results]
xgb_auc_gaps = [r['xgb_auc_gap'] for r in model_comparison_results]
avg_xgb_acc_gap = sum(xgb_acc_gaps) / len(xgb_acc_gaps)
avg_xgb_auc_gap = sum(xgb_auc_gaps) / len(xgb_auc_gaps)

print(f"\nAccuracy Gap (Train - Test):")
for i, gap in enumerate(xgb_acc_gaps, 1):
    status = "✅" if abs(gap) <= 0.05 else "⚠️ "
    print(f"  Window {i}: {gap:+.4f} {status}")
print(f"  Average: {avg_xgb_acc_gap:+.4f}")

print(f"\nROC AUC Gap (Train - Test):")
for i, gap in enumerate(xgb_auc_gaps, 1):
    status = "✅" if abs(gap) <= 0.05 else "⚠️ "
    print(f"  Window {i}: {gap:+.4f} {status}")
print(f"  Average: {avg_xgb_auc_gap:+.4f}")

print(f"\n{'INTERPRETATION - XGBOOST':^60}")
if avg_xgb_acc_gap < 0.05 and avg_xgb_auc_gap < 0.05:
    print("✅ NO OVERFITTING: Train/test gap < 5%")
    print("   Model generalizes well to unseen seasons")
elif avg_xgb_acc_gap < 0.10 and avg_xgb_auc_gap < 0.10:
    print("⚠️  MINIMAL OVERFITTING: Gap 5-10%")
    print("   Acceptable but monitor performance")
else:
    print("❌ SIGNIFICANT OVERFITTING: Gap > 10%")
    print("   Model may not generalize well")

# MLP Overfitting Analysis
print("\n" + "="*60)
print("OVERFITTING ANALYSIS - MLP")
print("="*60)

mlp_acc_gaps = [r['mlp_acc_gap'] for r in model_comparison_results]
mlp_auc_gaps = [r['mlp_auc_gap'] for r in model_comparison_results]
avg_mlp_acc_gap = sum(mlp_acc_gaps) / len(mlp_acc_gaps)
avg_mlp_auc_gap = sum(mlp_auc_gaps) / len(mlp_auc_gaps)

print(f"\nAccuracy Gap (Train - Test):")
for i, gap in enumerate(mlp_acc_gaps, 1):
    status = "✅" if abs(gap) <= 0.05 else "⚠️ "
    print(f"  Window {i}: {gap:+.4f} {status}")
print(f"  Average: {avg_mlp_acc_gap:+.4f}")

print(f"\nROC AUC Gap (Train - Test):")
for i, gap in enumerate(mlp_auc_gaps, 1):
    status = "✅" if abs(gap) <= 0.05 else "⚠️ "
    print(f"  Window {i}: {gap:+.4f} {status}")
print(f"  Average: {avg_mlp_auc_gap:+.4f}")

print(f"\n{'INTERPRETATION - MLP':^60}")
if avg_mlp_acc_gap < 0.05 and avg_mlp_auc_gap < 0.05:
    print("✅ NO OVERFITTING: Train/test gap < 5%")
    print("   Model generalizes well to unseen seasons")
elif avg_mlp_acc_gap < 0.10 and avg_mlp_auc_gap < 0.10:
    print("⚠️  MINIMAL OVERFITTING: Gap 5-10%")
    print("   Acceptable but monitor performance")
else:
    print("❌ SIGNIFICANT OVERFITTING: Gap > 10%")
    print("   Model may not generalize well")

print(f"\n{'DETAILED RESULTS - PER WINDOW':^60}")
print("\nPer-window breakdown:")
for i, result in enumerate(model_comparison_results, 1):
    print(f"\n  Window {i} (Test season: {result['test_season']}):")
    print(f"    XGBOOST:")
    print(f"      Train - Acc: {result['xgb_train_acc']:.4f}, AUC: {result['xgb_train_auc']:.4f}")
    print(f"      Test  - Acc: {result['xgb_test_acc']:.4f}, AUC: {result['xgb_test_auc']:.4f}")
    print(f"      Gap   - Acc: {result['xgb_acc_gap']:+.4f}, AUC: {result['xgb_auc_gap']:+.4f}")
    print(f"    MLP:")
    print(f"      Train - Acc: {result['mlp_train_acc']:.4f}, AUC: {result['mlp_train_auc']:.4f}")
    print(f"      Test  - Acc: {result['mlp_test_acc']:.4f}, AUC: {result['mlp_test_auc']:.4f}")
    print(f"      Gap   - Acc: {result['mlp_acc_gap']:+.4f}, AUC: {result['mlp_auc_gap']:+.4f}")

# Display predictions for last test window
print("\n" + "="*60)
print(f"PREDICTIONS FOR FINAL TEST WINDOW (Season {model_comparison_results[-1]['test_season']})")
print("="*60)
last_preds = all_predictions[-1]
print("XGBOOST Predictions (team, win_prob, winner):")
for i, row in last_preds.iterrows():
    print(f"  {row['Team']}: Prob={row['win_prob']:.3f}, Winner={int(row['winner'])}")

print("\nMLP Predictions (team, win_prob, winner):")
for i, row in last_preds.iterrows():
    print(f"  {row['Team']}: Prob={row['win_prob_mlp']:.3f}, Winner={int(row['winner'])}")

# Feature importance from XGBoost (MLP has no feature importance)
xgb_model_final = window_results[-1]['xgb_model']
print("\nTop features by XGBoost importance (from last window):")
importance = dict(zip(feature_cols, xgb_model_final.feature_importances_))
for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feat}: {imp:.4f}")

# ------- MONTE CARLO SEASON SIMULATION -------
print("\n" + "="*60)
print("MONTE CARLO SEASON SIMULATION")
print("="*60)# Extract data from final test window
final_season = window_results[-1]['test_season']
final_preds = all_predictions[-1].sort_values('win_prob', ascending=False).reset_index(drop=True)

# Get probabilities and teams
teams = final_preds['Team'].values
model_probs = final_preds['win_prob'].values

print(f"\nSeason: {final_season}")
print(f"Total teams: {len(teams)}")
print(f"Sum of model probabilities: {model_probs.sum():.4f}")

# Run Monte Carlo simulation (10,000 simulations)
empirical_probs, winner_counts = monte_carlo_season_simulation(
    teams=teams,
    win_probabilities=model_probs,
    num_simulations=10000
)

# Create comparison table
print("\n" + "="*60)
print("COMPARISON: MODEL PREDICTIONS vs EMPIRICAL SIMULATION")
print("="*60)

comparison_data = []
for i, team in enumerate(teams):
    comparison_data.append({
        'Team': team,
        'Model_Prob': model_probs[i],
        'Empirical_Prob': empirical_probs[team],
        'Win_Count': winner_counts[team],
        'Difference': empirical_probs[team] - model_probs[i]
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Empirical_Prob', ascending=False).reset_index(drop=True)
comparison_df.index = range(1, len(comparison_df) + 1)

print("\n{:<4} {:<25} {:<15} {:<15} {:<12} {:<12}".format(
    "Rank", "Team", "Model Prob", "Empirical Prob", "Win Count", "Difference"
))
print("-" * 95)

for idx, row in comparison_df.iterrows():
    print("{:<4} {:<25} {:<15.4f} {:<15.4f} {:<12} {:<12.4f}".format(
        idx,
        row['Team'][:24],
        row['Model_Prob'],
        row['Empirical_Prob'],
        int(row['Win_Count']),
        row['Difference']
    ))

print("-" * 95)
print(f"{'Total Empirical Prob:':<50} {comparison_df['Empirical_Prob'].sum():.4f}")
print(f"{'Total Model Prob:':<50} {comparison_df['Model_Prob'].sum():.4f}")

# Summary statistics
print("\n" + "="*60)
print("SIMULATION SUMMARY STATISTICS")
print("="*60)

prob_diff = comparison_df['Difference'].abs()
print(f"\nProbability Differences (|Model - Empirical|):")
print(f"  Mean absolute difference: {prob_diff.mean():.4f}")
print(f"  Max difference: {prob_diff.max():.4f}")
print(f"  Min difference: {prob_diff.min():.4f}")
print(f"  Std deviation: {prob_diff.std():.4f}")

# Find agreement/disagreement
strong_agreement = (comparison_df['Difference'].abs() < 0.02).sum()
moderate_agreement = (comparison_df['Difference'].abs() < 0.05).sum()
weak_agreement = (comparison_df['Difference'].abs() >= 0.05).sum()

print(f"\nAgreement between model and simulation:")
print(f"  Strong agreement (<2% diff): {strong_agreement} teams")
print(f"  Moderate agreement (<5% diff): {moderate_agreement} teams")
print(f"  Weak agreement (≥5% diff): {weak_agreement} teams")

# Identify most likely winners
print(f"\n" + "="*60)
print("MOST LIKELY WINNERS (By Empirical Simulation)")
print("="*60)

top_5 = comparison_df.head(5)
print(f"\nTop 5 teams by empirical win probability:")
for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
    pct = row['Empirical_Prob'] * 100
    print(f"  {rank}. {row['Team']:<25} {pct:>6.2f}% ({int(row['Win_Count']):,}/10,000 simulations)")
    print(f"     Model prediction: {row['Model_Prob']*100:>6.2f}% | Diff: {row['Difference']:+.4f}")

# Actual winner from test set
actual_winner_row = final_preds[final_preds['winner'] == 1]
if len(actual_winner_row) > 0:
    actual_winner = actual_winner_row.iloc[0]['Team']
    actual_prob = empirical_probs[actual_winner]
    print(f"\n✅ Actual league winner: {actual_winner}")
    print(f"   Empirical simulation probability: {actual_prob*100:.2f}%")
    actual_model_prob = comparison_df[comparison_df['Team'] == actual_winner]['Model_Prob'].values[0]
    print(f"   Model prediction: {actual_model_prob*100:.2f}%")
    
    # Check if correctly ranked
    rank = (comparison_df['Empirical_Prob'] > actual_prob).sum() + 1
    if rank == 1:
        print(f"   ✅ CORRECT: Model ranked actual winner 1st")
    else:
        print(f"   ⚠️  Model ranked actual winner {rank}th")

# Distribution analysis
print(f"\n" + "="*60)
print("EMPIRICAL DISTRIBUTION ANALYSIS")
print("="*60)

empirical_probs_list = comparison_df['Empirical_Prob'].values
print(f"\nEmpirical win probability distribution:")
print(f"  Mean: {empirical_probs_list.mean():.4f}")
print(f"  Median: {np.median(empirical_probs_list):.4f}")
print(f"  Std Dev: {empirical_probs_list.std():.4f}")
print(f"  Min: {empirical_probs_list.min():.4f}")
print(f"  Max: {empirical_probs_list.max():.4f}")
print(f"  Range: {empirical_probs_list.max() - empirical_probs_list.min():.4f}")

# Quantile analysis
quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
print(f"\nQuantile analysis:")
for q in quantiles:
    val = np.quantile(empirical_probs_list, q)
    print(f"  {q*100:.0f}th percentile: {val:.4f}")

print(f"\n{'='*60}")
print(f"Note: Empirical probabilities = frequency from {10000:,} simulations")
print(f"      Each simulation: full round-robin season with sampled match outcomes")
print(f"      Draw probability: 25% (empirical EPL average)")
print(f"{'='*60}")

# ========== 2026 SEASON PREDICTION ==========
print("\n" + "="*60)
print("2026 PREMIER LEAGUE SEASON PREDICTION")
print("="*60)

# Check if we have data for 2023, 2024, 2025
available_seasons_for_2026 = sorted([s for s in seasons_sorted if int(s) >= 2023])

if len(available_seasons_for_2026) >= 3:
    print(f"\n✅ Found {len(available_seasons_for_2026)} recent seasons for training: {available_seasons_for_2026}")
    
    # Use the last 3 seasons (or all available if fewer than 3)
    train_seasons_2026 = sorted(seasons_sorted)[-3:] if len(seasons_sorted) >= 3 else sorted(seasons_sorted)
    print(f"Training on seasons: {train_seasons_2026}")
    
    # Prepare training data
    df_train_2026 = df[df["season"].isin(train_seasons_2026)].copy()
    
    X_train_2026 = df_train_2026[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_train_2026 = df_train_2026["winner"].values
    
    # Scale features
    scaler_2026 = StandardScaler()
    X_train_2026 = scaler_2026.fit_transform(X_train_2026)
    
    # Train XGBoost on all recent data for 2026 prediction
    xgb_2026 = xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        verbose=0,
        random_state=42
    )
    xgb_2026.fit(X_train_2026, y_train_2026)
    
    print(f"\n✅ XGBoost model trained on {len(X_train_2026)} samples ({len(train_seasons_2026)} seasons)")
    
    # Get all teams from the most recent season
    latest_season = max(seasons_sorted)
    df_latest = df[df["season"] == latest_season].copy()
    teams_2026 = df_latest['Team'].values
    
    print(f"✅ Retrieved {len(teams_2026)} teams from season {latest_season}")
    
    # Prepare data for prediction
    X_latest = df_latest[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_latest_scaled = scaler_2026.transform(X_latest)
    
    # Generate predictions
    proba_2026 = xgb_2026.predict_proba(X_latest_scaled)[:, 1]
    
    # Normalize probabilities to sum to 1 (accounting for randomness)
    proba_2026_norm = proba_2026 / proba_2026.sum()
    
    # Create results dataframe
    predictions_2026 = pd.DataFrame({
        'Rank': range(1, len(teams_2026) + 1),
        'Team': teams_2026,
        'Raw_Probability': proba_2026,
        'Normalized_Probability': proba_2026_norm,
        'Percentage': proba_2026_norm * 100
    })
    
    # Sort by probability
    predictions_2026 = predictions_2026.sort_values('Normalized_Probability', ascending=False).reset_index(drop=True)
    predictions_2026['Rank'] = range(1, len(predictions_2026) + 1)
    
    print(f"\n{'='*80}")
    print(f"{'2026 PREMIER LEAGUE WINNER PREDICTION - RANKED PROBABILITY TABLE':^80}")
    print(f"{'='*80}")
    print(f"Training data: Seasons {train_seasons_2026} | Model: XGBoost (100 trees)")
    print(f"Current teams: {len(teams_2026)} | Base season: {latest_season}")
    print(f"{'='*80}\n")
    
    # Display full table
    print(f"{'Rank':<6} {'Team':<25} {'Probability':<15} {'Percentage':<12} {'Odds':<12}")
    print(f"{'-'*80}")
    
    for idx, row in predictions_2026.iterrows():
        rank = row['Rank']
        team = row['Team']
        prob = row['Normalized_Probability']
        pct = row['Percentage']
        
        # Calculate implied odds (decimal)
        if prob > 0:
            decimal_odds = 1.0 / prob
            fractional = f"{decimal_odds:.2f}:1"
        else:
            fractional = "N/A"
        
        rank_str = f"{rank:<6}"
        team_str = f"{team:<25}"
        prob_str = f"{prob:.6f}    "
        pct_str = f"{pct:>6.2f}%   "
        odds_str = f"{fractional:<12}"
        
        print(f"{rank_str}{team_str}{prob_str}{pct_str}{odds_str}")
    
    print(f"{'-'*80}")
    
    # Top 5 Analysis
    print(f"\n{'TOP 5 FAVORITES FOR 2026':^80}")
    print(f"{'-'*80}\n")
    
    top_5 = predictions_2026.head(5)
    for idx, row in top_5.iterrows():
        rank = row['Rank']
        team = row['Team']
        pct = row['Percentage']
        prob = row['Normalized_Probability']
        
        # Visual representation
        bar_length = int(pct / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"{rank}. {team:<22} {pct:>6.2f}% [{bar}]")
        print(f"   Probability: {prob:.6f}\n")
    
    # Statistical Analysis
    print(f"\n{'STATISTICAL ANALYSIS':^80}")
    print(f"{'-'*80}\n")
    
    probs = predictions_2026['Normalized_Probability'].values
    percentages = predictions_2026['Percentage'].values
    
    print(f"Mean probability:       {probs.mean():.6f} ({percentages.mean():.2f}%)")
    print(f"Median probability:     {np.median(probs):.6f} ({np.median(percentages):.2f}%)")
    print(f"Std deviation:          {probs.std():.6f} ({percentages.std():.2f}%)")
    print(f"Min probability:        {probs.min():.6f} ({percentages.min():.2f}%)")
    print(f"Max probability:        {probs.max():.6f} ({percentages.max():.2f}%)")
    print(f"Range:                  {probs.max() - probs.min():.6f} ({percentages.max() - percentages.min():.2f}%)")
    
    # Concentration analysis
    top_3_prob = predictions_2026.head(3)['Normalized_Probability'].sum()
    top_5_prob = predictions_2026.head(5)['Normalized_Probability'].sum()
    top_10_prob = predictions_2026.head(10)['Normalized_Probability'].sum()
    
    print(f"\nConcentration of probability:")
    print(f"  Top 3 teams:   {top_3_prob*100:>6.2f}% combined")
    print(f"  Top 5 teams:   {top_5_prob*100:>6.2f}% combined")
    print(f"  Top 10 teams:  {top_10_prob*100:>6.2f}% combined")
    print(f"  Rest ({len(predictions_2026)-10} teams): {(1-top_10_prob)*100:>6.2f}% combined")
    
    # Save 2026 predictions to file
    predictions_2026.to_csv('2026_predictions.csv', index=False)
    print(f"\n✅ Predictions saved to: 2026_predictions.csv")
    
    print(f"\n{'='*80}")
    print(f"Disclaimer: These predictions are based on historical EPL data (seasons {train_seasons_2026})")
    print(f"and assume team compositions remain similar to season {latest_season}.")
    print(f"Real-world factors (transfers, injuries, etc.) not accounted for.")
    print(f"{'='*80}")

elif len(available_seasons_for_2026) > 0:
    print(f"\n⚠️  Only {len(available_seasons_for_2026)} recent season(s) available: {available_seasons_for_2026}")
    print(f"Cannot create rolling window (need minimum 4 seasons: 3 train + 1 test)")
    print(f"Proceeding with available data...")
    
    # Train on whatever we have
    train_seasons_2026 = available_seasons_for_2026
    df_train_2026 = df[df["season"].isin(train_seasons_2026)].copy()
    
    X_train_2026 = df_train_2026[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_train_2026 = df_train_2026["winner"].values
    
    scaler_2026 = StandardScaler()
    X_train_2026 = scaler_2026.fit_transform(X_train_2026)
    
    xgb_2026 = xgb.XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        verbose=0,
        random_state=42
    )
    xgb_2026.fit(X_train_2026, y_train_2026)
    
    print(f"✅ XGBoost trained on {len(X_train_2026)} samples from {len(train_seasons_2026)} available season(s)")
    
    # Predict for latest season teams
    latest_season = max(seasons_sorted)
    df_latest = df[df["season"] == latest_season].copy()
    teams_2026 = df_latest['Team'].values
    
    X_latest = df_latest[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_latest_scaled = scaler_2026.transform(X_latest)
    
    proba_2026 = xgb_2026.predict_proba(X_latest_scaled)[:, 1]
    proba_2026_norm = proba_2026 / proba_2026.sum()
    
    predictions_2026 = pd.DataFrame({
        'Team': teams_2026,
        'Probability': proba_2026_norm,
        'Percentage': proba_2026_norm * 100
    }).sort_values('Probability', ascending=False).reset_index(drop=True)
    
    print(f"\n{'2026 PREDICTION (LIMITED DATA)':^60}")
    print(f"{'='*60}\n")
    
    for idx, row in predictions_2026.head(10).iterrows():
        print(f"{idx+1}. {row['Team']:<22} {row['Percentage']:>6.2f}%")

else:
    print(f"\n❌ No data available beyond 2021. Predictions for 2026 not possible.")
    print(f"Available seasons: {seasons_sorted}")
    print(f"Please update data source to include seasons 2023-2025.")

# ========== IN-SEASON UPDATE: Use current 2026 standings (if available) ==========
print('\n' + '='*60)
print('IN-SEASON UPDATE: Checking for current 2026 standings')
print('='*60)

# Try to fetch current 2026 standings via Understat
try:
    current_season_data = asyncio.run(fetch_understat_data([2026]))
except Exception as e:
    current_season_data = []
    print(f"  Unable to fetch 2026 standings: {e}")

if current_season_data and len(current_season_data) > 0:
    print(f"  ✅ Fetched current 2026 standings ({len(current_season_data)} team records). Updating probabilities...")
    df_current = pd.DataFrame(current_season_data)
    # Ensure feature columns exist and are numeric
    for col in ['G', 'GA', 'M', 'xG', 'xGA', 'PTS']:
        if col in df_current.columns:
            df_current[col] = pd.to_numeric(df_current[col], errors='coerce').fillna(0)
        else:
            df_current[col] = 0

    # Engineer same features: GD, PPM, xGD
    df_current['GD'] = df_current['G'] - df_current['GA']
    df_current['PPM'] = df_current['PTS'] / df_current['M'].replace(0, 1)
    df_current['xGD'] = pd.to_numeric(df_current.get('xG', 0), errors='coerce').fillna(0) - pd.to_numeric(df_current.get('xGA', 0), errors='coerce').fillna(0)

    # Ensure all feature_cols exist in df_current (fill missing with 0)
    for c in feature_cols:
        if c not in df_current.columns:
            df_current[c] = 0

    # Prepare X and scale using the scaler trained on 2023-2025 (scaler_2026)
    X_current = df_current[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    try:
        X_current_scaled = scaler_2026.transform(X_current)
    except Exception as e:
        # If scaler_2026 not defined (edge case), fit a new scaler on X_train_2026
        print(f"  Warning: scaler_2026 not available ({e}), refitting scaler on last training data")
        scaler_2026 = StandardScaler()
        X_train_2026 = df_train_2026[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        scaler_2026.fit(X_train_2026)
        X_current_scaled = scaler_2026.transform(X_current)

    # Predict probabilities for current standings
    proba_current = xgb_2026.predict_proba(X_current_scaled)[:, 1]
    proba_current_norm = proba_current / proba_current.sum()

    df_current_preds = pd.DataFrame({
        'Team': df_current['Team'].values,
        'InSeason_Prob': proba_current,
        'InSeason_Prob_Norm': proba_current_norm,
        'InSeason_Pct': proba_current_norm * 100
    }).sort_values('InSeason_Prob_Norm', ascending=False).reset_index(drop=True)

    # Merge with pre-season predictions (predictions_2026)
    pre_season_df = predictions_2026[['Team', 'Normalized_Probability']].rename(columns={'Normalized_Probability': 'PreSeason_Prob'})
    compare_df = pre_season_df.merge(df_current_preds[['Team', 'InSeason_Prob_Norm']], on='Team', how='outer').fillna(0)
    compare_df['PreSeason_Pct'] = compare_df['PreSeason_Prob'] * 100
    compare_df['InSeason_Pct'] = compare_df['InSeason_Prob_Norm'] * 100
    compare_df['Diff_Pct'] = compare_df['InSeason_Pct'] - compare_df['PreSeason_Pct']

    # Run Monte Carlo on in-season probabilities
    teams_curr = df_current['Team'].values
    probs_for_sim = proba_current_norm
    print('\n  Running Monte Carlo with in-season probabilities (10,000 simulations)...')
    empirical_curr, winner_counts_curr = monte_carlo_season_simulation(teams=teams_curr, win_probabilities=probs_for_sim, num_simulations=10000)

    # Add empirical probabilities to compare_df
    compare_df['Empirical_Prob'] = compare_df['Team'].apply(lambda t: empirical_curr.get(t, 0))
    compare_df = compare_df.sort_values('InSeason_Prob_Norm', ascending=False).reset_index(drop=True)

    # Print comparison summary
    print('\n  Pre-season vs In-season Top 10 changes:')
    print('  Rank | Team                      | Pre%    | In-Season% | Delta%   | Empirical%')
    print('  ---- | ------------------------- | ------- | ---------- | -------- | ----------')
    for i, row in compare_df.head(10).iterrows():
        print(f"  {i+1:>3}  | {row['Team']:<25} | {row['PreSeason_Pct']:7.2f}% | {row['InSeason_Pct']:9.2f}% | {row['Diff_Pct']:8.2f}% | {row['Empirical_Prob']*100:9.2f}%")

    # Save compare table
    compare_df.to_csv('2026_pre_vs_inseason_comparison.csv', index=False)
    print('\n  ✅ Saved comparison to 2026_pre_vs_inseason_comparison.csv')

else:
    print('  ⚠️  No current 2026 standings found via API. Cannot update in-season probabilities.')

# In-season update completed (todo recorded in agent session)
print('\n  In-season update: comparison complete.')

