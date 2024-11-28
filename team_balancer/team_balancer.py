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

    # Create a DataFrame for new players with default stats
    new_players_data = []
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
        new_players_data.append(default_stats)

    # Convert to DataFrame and concatenate with the existing player stats
    if new_players_data:
        new_players_df = pd.DataFrame(new_players_data)
        player_stats = pd.concat([player_stats, new_players_df], ignore_index=True)

    return player_stats

# Function to generate balanced teams with random additional players
def generate_balanced_teams(players, player_stats, total_players):
    # Filter out players already in the list
    available_players = player_stats[~player_stats["Player"].isin(players)]

    # Randomly select additional players to fill up the required total
    additional_players_needed = total_players - len(players)
    if additional_players_needed > 0:
        additional_players = available_players.sample(n=additional_players_needed, replace=False)["Player"].tolist()
        players.extend(additional_players)

    # Create the final player dataset
    player_data = player_stats[player_stats["Player"].isin(players)].copy()
    player_data["Strength"] = player_data.apply(calculate_strength, axis=1)

    # Sort players by strength to distribute them more evenly
    sorted_players = player_data.sort_values(by="Strength", ascending=False).reset_index(drop=True)

    # Assign players alternately to teams to balance strength
    team_a, team_b = [], []
    for i, row in sorted_players.iterrows():
        if i % 2 == 0:
            team_a.append(row)
        else:
            team_b.append(row)

    # Convert lists to DataFrames
    team_a = pd.DataFrame(team_a)
    team_b = pd.DataFrame(team_b)

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

    features = pd.DataFrame([[team1_strength, team2_strength, team1_defense, team2_attack]],
                            columns=["Team1_Strength", "Team2_Strength", "Team1_Defense", "Team2_Attack"])
    prediction = model.predict(features)[0]
    winner = "Team A Wins" if prediction > 0.5 else "Team B Wins"
    return prediction, winner

# Function to simulate multiple runs
def simulate_runs(players, player_stats, model, runs=10, team_size=7):
    team_a_wins = 0
    team_b_wins = 0
    total_players = team_size * 2  # Calculate the total number of players needed

    results = []
    for i in range(runs):
        team_a, team_b = generate_balanced_teams(players, player_stats, total_players)
        prediction, winner = predict_outcome(model, team_a, team_b)
        if prediction is None:
            break

        results.append((i + 1, prediction, winner, team_a, team_b))

        if winner == "Team A Wins":
            team_a_wins += 1
        else:
            team_b_wins += 1

    # Print simulation summary
    print(f"\nSimulation Results ({runs} Runs):")
    print(f"Team A Wins: {team_a_wins}")
    print(f"Team B Wins: {team_b_wins}")

    # Plot results
    x = list(range(1, runs + 1))
    y = [result[1] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Predicted Outcome (Team A Wins)")
    plt.axhline(0.5, color="r", linestyle="--", label="50% Threshold")
    plt.title("Prediction Across Multiple Simulations")
    plt.xlabel("Simulation #")
    plt.ylabel("Predicted Outcome (Team A Probability)")
    plt.legend()

    # Annotate player lists for one sample
    if results:
        team_a_players = ", ".join(results[0][3]["Player"].tolist())
        team_b_players = ", ".join(results[0][4]["Player"].tolist())
        plt.text(1, 0.8, f"Team A: {team_a_players}", fontsize=10, color="blue")
        plt.text(1, 0.7, f"Team B: {team_b_players}", fontsize=10, color="green")

    plt.show()

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate balanced soccer teams.")
    parser.add_argument(
        "team_size",
        type=int,
        choices=[7, 8],
        help="Number of players per team (7 or 8).",
    )
    parser.add_argument(
        "players", 
        nargs="+", 
        help="List of player names for the week (space-separated)"
    )
    args = parser.parse_args()
    team_size = args.team_size
    players = args.players

    total_required_players = team_size * 2

    # Load player stats
    player_stats = load_player_stats()

    # Handle missing players in stats
    player_stats = handle_missing_players(players, player_stats)

    # Generate teams, filling in additional players if needed
    team_a, team_b = generate_balanced_teams(players, player_stats, total_required_players)

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
    # Save teams to CSV
    team_output.to_csv(TEAMS_FILE, index=False)
    print(f"Teams saved to {TEAMS_FILE}")

    # Print teams neatly
    team_a = team_output[team_output["Team"] == "Team A"]
    team_b = team_output[team_output["Team"] == "Team B"]

    print("\nTeam Composition:\n")
    print("Team A:")
    for _, row in team_a.iterrows():
        print(f"  - {row['Player']} (Role: {row['Player']})")

    print("\nTeam B:")
    for _, row in team_b.iterrows():
        print(f"  - {row['Player']} (Role: {row['Player']})")


    # Simulate multiple runs and visualize results
    simulate_runs(players, player_stats, model, runs=10, team_size=team_size)

    # Save updated player stats
    save_player_stats(player_stats)


if __name__ == "__main__":
    main()
