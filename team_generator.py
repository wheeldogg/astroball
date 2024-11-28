import pandas as pd
import random
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

# File paths
PLAYER_STATS_FILE = "data/player_stats.csv"
HISTORICAL_RESULTS_FILE = "data/historical_results.csv"
TEAMS_FILE = "data/teams.csv"

# Function to load or create player stats
def load_player_stats():
    if os.path.exists(PLAYER_STATS_FILE):
        return pd.read_csv(PLAYER_STATS_FILE)
    else:
        columns = [
            "Player", "Defense", "Attack", "Versatility", "Stamina", "Goals",
            "Assists", "Wins", "Losses", "MVP", "Attendance"
        ]
        return pd.DataFrame(columns=columns)

# Function to save player stats
def save_player_stats(player_stats):
    player_stats.to_csv(PLAYER_STATS_FILE, index=False)

# Function to calculate player strength
def calculate_strength(row):
    return (
        row["Defense"] * 0.3 +
        row["Attack"] * 0.4 +
        row["Versatility"] * 0.2 +
        row["Stamina"] * 0.1 +
        row["Goals"] * 0.5 +
        row["Assists"] * 0.3 +
        row["Wins"] * 0.2 -
        row["Losses"] * 0.1
    )

# Function to handle missing players
def handle_missing_players(players, player_stats):
    existing_players = player_stats["Player"].tolist()
    new_players = [p for p in players if p not in existing_players]

    # Add missing players with default stats
    for player in new_players:
        default_stats = {
            "Player": player,
            "Defense": random.randint(2, 4),
            "Attack": random.randint(2, 4),
            "Versatility": random.randint(2, 4),
            "Stamina": random.randint(2, 4),
            "Goals": 0,
            "Assists": 0,
            "Wins": 0,
            "Losses": 0,
            "MVP": 0,
            "Attendance": 0,
        }
        player_stats = player_stats.append(default_stats, ignore_index=True)

    return player_stats

# Function to generate balanced teams
def generate_balanced_teams(players, player_stats):
    player_data = player_stats[player_stats["Player"].isin(players)].copy()
    player_data["Strength"] = player_data.apply(calculate_strength, axis=1)

    # Shuffle players and split into two teams
    shuffled_players = player_data.sample(frac=1).reset_index(drop=True)
    team_a = shuffled_players.iloc[:len(players) // 2]
    team_b = shuffled_players.iloc[len(players) // 2:]

    return team_a, team_b

# Function to train the prediction model
def train_model():
    if os.path.exists(HISTORICAL_RESULTS_FILE):
        historical_data = pd.read_csv(HISTORICAL_RESULTS_FILE)

        X = historical_data[["Team1_Strength", "Team2_Strength", "Team1_Defense", "Team2_Attack"]]
        y = historical_data["Outcome"]  # 1 = Team 1 wins, 0 = Team 2 wins

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"Model trained. Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        return model
    else:
        print("No historical data found. Skipping model training.")
        return None

# Function to predict match outcome
def predict_outcome(model, team_a, team_b):
    if not model:
        return None, "Prediction unavailable (no model trained)."

    team1_strength = team_a["Strength"].sum()
    team2_strength = team_b["Strength"].sum()
    team1_defense = team_a["Defense"].sum()
    team2_attack = team_b["Attack"].sum()

    prediction = model.predict([[team1_strength, team2_strength, team1_defense, team2_attack]])
    winner = "Team A Wins" if prediction > 0.5 else "Team B Wins"
    return prediction, winner

# Function to simulate multiple runs
def simulate_runs(players, player_stats, model, runs=10):
    team_a_wins = 0
    team_b_wins = 0

    for i in range(runs):
        team_a, team_b = generate_balanced_teams(players, player_stats)
        prediction, winner = predict_outcome(model, team_a, team_b)
        if prediction is None:
            break
        if winner == "Team A Wins":
            team_a_wins += 1
        else:
            team_b_wins += 1

    # Print valuable information
    print(f"\nSimulation Results ({runs} Runs):")
    print(f"Team A Wins: {team_a_wins}")
    print(f"Team B Wins: {team_b_wins}")

    # Generate line graph
    win_percentages = [team_a_wins / runs * 100, team_b_wins / runs * 100]
    plt.plot(["Team A", "Team B"], win_percentages, marker="o")
    plt.axhline(50, color="r", linestyle="--", label="Expected Fairness Line (50%)")
    plt.title("Team Win Percentages Across Simulations")
    plt.ylabel("Win Percentage")
    plt.legend()
    plt.show()

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate balanced soccer teams.")
    parser.add_argument(
        "players", 
        nargs="+", 
        help="List of player names for the week (space-separated)"
    )
    args = parser.parse_args()
    players = args.players

    # Load player stats
    player_stats = load_player_stats()

    # Handle missing players
    player_stats = handle_missing_players(players, player_stats)

    # Generate teams
    team_a, team_b = generate_balanced_teams(players, player_stats)

    # Train the model
    model = train_model()

    # Predict match outcome
    prediction, outcome = predict_outcome(model, team_a, team_b)

    # Print detailed output
    print("\nDetailed Output for UI Development:")
    print(f"Team A Strength: {team_a['Strength'].sum()}")
    print(f"Team B Strength: {team_b['Strength'].sum()}")
    print(f"Team A Defense: {team_a['Defense'].sum()}")
    print(f"Team B Attack: {team_b['Attack'].sum()}")
    print(f"Predicted Outcome: {outcome}")

    # Save teams to CSV
    team_output = pd.concat([
        team_a.assign(Team="Team A"), 
        team_b.assign(Team="Team B")
    ])
    team_output.to_csv(TEAMS_FILE, index=False)
    print(f"Teams saved to {TEAMS_FILE}")

    # Simulate multiple runs and visualize results
    simulate_runs(players, player_stats, model, runs=10)

    # Save updated player stats
    save_player_stats(player_stats)

if __name__ == "__main__":
    main()
