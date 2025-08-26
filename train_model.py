import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- LEAGUE MAPPING (Defines the files this script can process) ---
LEAGUES = {
    "1": {"name": "English Premier League", "file_key": "premier_league"},
    "2": {"name": "Spanish La Liga", "file_key": "la_liga"},
    "3": {"name": "German Bundesliga", "file_key": "bundesliga"},
    # You can add Serie A here later when you have the data
    # "4": {"name": "Italian Serie A", "file_key": "serie_a"}, 
}

def create_features(df):
    """Engineers features from the raw match data for machine learning."""
    print("Engineering features from the dataset...")
    df = df.sort_values('date')

    # Data cleaning and type conversion
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
    df.dropna(subset=['home_goals', 'away_goals'], inplace=True)
    df['home_goals'] = df['home_goals'].astype(int)
    df['away_goals'] = df['away_goals'].astype(int)
    
    # Map team names to unique integer IDs for calculations
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_map = {team: i for i, team in enumerate(all_teams)}
    df['home_team_id'] = df['home_team'].map(team_map)
    df['away_team_id'] = df['away_team'].map(team_map)

    # Calculate rolling averages for form (goals, etc.) over the last 5 games
    df['h_avg_goals'] = df.groupby('home_team_id')['home_goals'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['a_avg_goals'] = df.groupby('away_team_id')['away_goals'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['h_avg_conceded'] = df.groupby('home_team_id')['away_goals'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['a_avg_conceded'] = df.groupby('away_team_id')['home_goals'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    # Calculate form points (Win=3, Draw=1, Loss=0)
    def get_points(row, team_type):
        if (team_type == 'home' and row['result'] == 1) or (team_type == 'away' and row['result'] == 2): return 3
        if row['result'] == 0: return 1
        return 0
    df['h_points'] = df.apply(lambda row: get_points(row, 'home'), axis=1)
    df['a_points'] = df.apply(lambda row: get_points(row, 'away'), axis=1)
    df['h_form_pts'] = df.groupby('home_team_id')['h_points'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    df['a_form_pts'] = df.groupby('away_team_id')['a_points'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    
    # Finalize the dataset for training, dropping rows with NaN values
    features_df = df[['h_avg_goals', 'a_avg_goals', 'h_avg_conceded', 'a_avg_conceded', 'h_form_pts', 'a_form_pts', 'result']].dropna()
    return features_df

def main():
    """Main function to load a dataset, train a model, and save it."""
    # --- Interactive League Selection ---
    print("Please choose a league to train a model for:")
    for key, league_info in LEAGUES.items():
        print(f"  [{key}] {league_info['name']}")
    
    choice = input("Enter the number of your choice: ")
    
    if choice not in LEAGUES:
        print("Invalid choice. Exiting.")
        return
        
    selected_league = LEAGUES[choice]
    input_csv_file = f"{selected_league['file_key']}_raw_data.csv"
    output_model_file = f"trained_model_{selected_league['file_key']}.pkl"
    print(f"\nYou have selected: {selected_league['name']}\n")

    if not os.path.exists(input_csv_file):
        print(f"FATAL ERROR: The data file '{input_csv_file}' was not found.")
        print(f"Please create the CSV file with the historical data for this league.")
        return

    print(f"Loading raw data from {input_csv_file}...")
    df = pd.read_csv(input_csv_file)

    training_data = create_features(df)
    
    X = training_data.drop('result', axis=1)
    y = training_data['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining Machine Learning model (RandomForestClassifier)...")
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features='sqrt', random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained successfully!")
    print(f"Accuracy on unseen test data for {selected_league['name']}: {accuracy*100:.2f}%")

    joblib.dump(model, output_model_file)
    print(f"✅ Model saved to: {output_model_file}. The main app can now use this model.")

if __name__ == "__main__":
    main()