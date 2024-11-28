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

# Globals
VERBOSE = False

# Function to load or create player stats
def load_player_stats():
    if os.path.exists(PLAYER_STATS_FILE):
        return pd.read_csv(PLAYER_STATS_FILE)
    else:
        columns = ["Overall", "Defense", "Attack", "Stamina", "Versatility", "Role"]
        return pd.DataFrame(columns=columns)

# Function to save player stats
def save_player_stats(player_stats):
    player_stats.to_csv(PLAYER_STATS_FILE, index=False)

def calculate_strength(row):
    """
    Calculate player strength using weighted attributes.
    Adjust the weights based on importance and the data ranges.
    """
    return (
        row["Overall"] * 0.9 +  # Overall is a significant contributor
        row["Defense"] * 0.25 +  # Defense is important but secondary
        row["Attack"] * 0.25 +   # Attack contributes similarly to Defense
        row["Stamina"] * 0.25 +  # Stamina plays a moderate role
        row["Versatility"] * 0.25  # Versatility is a minor factor
    )

# Function to handle missing players
def handle_missing_players(players, player_stats):
    existing_players = player_stats["Role"].tolist()
    new_players = [p for p in players if p not in existing_players]

    # Create a DataFrame for new players with default stats
    new_players_data = []
    roles = ["Defender", "Midfielder", "Striker"]
    for player in new_players:
        default_stats = {
            "Overall": random.randint(50, 70),
            "Defense": random.randint(2, 4),
            "Attack": random.randint(2, 4),
            "Stamina": random.randint(2, 4),
            "Versatility": random.randint(2, 4),
            "Role": player,  # Assign player's name as Role
        }
        new_players_data.append(default_stats)

    # Convert to DataFrame and concatenate with the existing player stats
    if new_players_data:
        new_players_df = pd.DataFrame(new_players_data)
        player_stats = pd.concat([player_stats, new_players_df], ignore_index=True)

    return player_stats

def balance_teams(team_a, team_b, strength_threshold=10, protected_players=None):
    """
    Swap players between Team A and Team B to balance their strengths.
    Ensure the strength difference is within the specified threshold.
    Do not swap out protected players.
    """
    if protected_players is None:
        protected_players = []

    max_iterations = 100  # Prevent infinite loops

    for _ in range(max_iterations):
        # Calculate current team strengths
        strength_a = team_a["Strength"].sum()
        strength_b = team_b["Strength"].sum()
        difference = abs(strength_a - strength_b)

        # If the difference is within the acceptable range, stop balancing
        if difference <= strength_threshold:
            break

        # Identify candidates for swapping, excluding protected players
        if strength_a > strength_b:
            # Team A is stronger
            team_a_candidates = team_a[~team_a["Player"].isin(protected_players)]
            team_b_candidates = team_b[~team_b["Player"].isin(protected_players)]
            if team_a_candidates.empty or team_b_candidates.empty:
                break  # No candidates available for swapping
            player_to_swap_out = team_a_candidates.loc[team_a_candidates["Strength"].idxmin()]
            player_to_swap_in = team_b_candidates.loc[team_b_candidates["Strength"].idxmax()]
        else:
            # Team B is stronger
            team_b_candidates = team_b[~team_b["Player"].isin(protected_players)]
            team_a_candidates = team_a[~team_a["Player"].isin(protected_players)]
            if team_b_candidates.empty or team_a_candidates.empty:
                break  # No candidates available for swapping
            player_to_swap_out = team_b_candidates.loc[team_b_candidates["Strength"].idxmin()]
            player_to_swap_in = team_a_candidates.loc[team_a_candidates["Strength"].idxmax()]

        # Perform the swap
        if strength_a > strength_b:
            # Swap weakest in Team A with strongest in Team B
            team_a = team_a.drop(player_to_swap_out.name)
            team_b = team_b.drop(player_to_swap_in.name)
            team_a = pd.concat([team_a, player_to_swap_in.to_frame().T], ignore_index=True)
            team_b = pd.concat([team_b, player_to_swap_out.to_frame().T], ignore_index=True)
        else:
            # Swap weakest in Team B with strongest in Team A
            team_b = team_b.drop(player_to_swap_out.name)
            team_a = team_a.drop(player_to_swap_in.name)
            team_b = pd.concat([team_b, player_to_swap_in.to_frame().T], ignore_index=True)
            team_a = pd.concat([team_a, player_to_swap_out.to_frame().T], ignore_index=True)

    # Final check
    final_difference = abs(team_a["Strength"].sum() - team_b["Strength"].sum())
    if VERBOSE:
        print(f"Final strength difference after balancing: {final_difference:.1f}")
    return team_a, team_b


def generate_balanced_teams(players, player_stats, total_players):
    """
    Generate balanced teams while ensuring that the provided players are always included
    and protected during balancing.
    """
    # Ensure we have enough players by adding additional ones
    available_players = player_stats[~player_stats["Player"].isin(players)]
    additional_players_needed = total_players - len(players)

    if additional_players_needed > 0:
        # Sample additional players to fill the required total players
        additional_players = available_players.sample(
            n=additional_players_needed, replace=False
        )["Player"].tolist()
    else:
        additional_players = []

    # Combine provided players with additional players
    current_players = players + additional_players  # Do not modify the original `players` list

    # Create the final player dataset
    player_data = player_stats[player_stats["Player"].isin(current_players)].copy()
    player_data["Strength"] = player_data.apply(calculate_strength, axis=1)

    # Separate protected players and remaining players
    protected_data = player_data[player_data["Player"].isin(players)].copy()
    remaining_data = player_data[~player_data["Player"].isin(players)].copy()

    # Shuffle remaining players for fair distribution
    remaining_data = remaining_data.sample(frac=1).reset_index(drop=True)

    # Initialize teams
    team_a = []
    team_b = []

    # Assign protected players alternately to teams
    for _, row in protected_data.iterrows():
        if len(team_a) <= len(team_b):
            team_a.append(row)
        else:
            team_b.append(row)

    # Assign remaining players alternately to teams
    for _, row in remaining_data.iterrows():
        if len(team_a) < total_players // 2:
            team_a.append(row)
        elif len(team_b) < total_players // 2:
            team_b.append(row)

    # Convert lists to DataFrames
    team_a = pd.DataFrame(team_a)
    team_b = pd.DataFrame(team_b)

    # Assert that all provided players are included in the final teams
    combined_players = pd.concat([team_a['Player'], team_b['Player']]).tolist()
    missing_players = [player for player in players if player not in combined_players]
    assert not missing_players, f"The following players are missing from the final teams: {missing_players}"

    # Post-assignment balancing with protected players
    team_a, team_b = balance_teams(team_a, team_b, strength_threshold=1000, protected_players=players)

    # Assert again after balancing
    combined_players = pd.concat([team_a['Player'], team_b['Player']]).tolist()
    missing_players = [player for player in players if player not in combined_players]
    assert not missing_players, f"Post Balancing: The following players are missing from the final teams: {missing_players}"

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

def simulate_runs(players, player_stats, model, runs=10, team_size=7):
    team_a_wins = 0
    team_b_wins = 0
    total_players = team_size * 2  # Calculate the total number of players needed

    results = []
    most_balanced = {"difference": float("inf"), "team_a": None, "team_b": None}  # Track most balanced teams

    for i in range(runs):
        # Generate balanced teams for each run
        team_a, team_b = generate_balanced_teams(players, player_stats, total_players)
        # Predict outcome
        prediction, winner = predict_outcome(model, team_a, team_b)
        if prediction is None:
            break

        # Calculate team strengths and difference
        strength_a = team_a["Strength"].sum()
        strength_b = team_b["Strength"].sum()
        difference = abs(strength_a - strength_b)
        if VERBOSE:
            print(f"Run {i+1}: Team A Strength = {strength_a:.1f}, Team B Strength = {strength_b:.1f}, Difference = {difference:.1f}")

        # Update most balanced teams if this permutation is closer
        if difference < most_balanced["difference"]:
            most_balanced["difference"] = difference
            most_balanced["team_a"] = team_a.copy()
            most_balanced["team_b"] = team_b.copy()

        # Collect results for analysis and plotting
        results.append((i + 1, prediction, winner, team_a, team_b))

        # Track wins
        if winner == "Team A Wins":
            team_a_wins += 1
        else:
            team_b_wins += 1

    # Print simulation summary
    print(f"\nSimulation Results ({runs} Runs):")
    print(f"Team A Wins: {team_a_wins}")
    print(f"Team B Wins: {team_b_wins}")

    print(type(most_balanced))

    # Display the most balanced teams
    print("\nMost Balanced Team Composition:")

    # most_balanced_df = pd.DataFrame(most_balanced)
    # print(most_balanced["team_a"])
    # print(most_balanced["team_b"])

    if most_balanced["team_a"] is not None and most_balanced["team_b"] is not None:
        # Ensure we use the correct columns
        team_a_summary = most_balanced["team_a"].copy()
        team_b_summary = most_balanced["team_b"].copy()

        # Add 'Player' explicitly if it does not already exist
        if "Player" not in team_a_summary.columns:
            team_a_summary["Player"] = team_a_summary.index  # Use index or other source for names
        if "Player" not in team_b_summary.columns:
            team_b_summary["Player"] = team_b_summary.index

        # Reorder columns to show 'Player' alongside their attributes
        team_a_summary = team_a_summary[["Player", "Role", "Overall", "Defense", "Attack", "Strength"]]
        team_b_summary = team_b_summary[["Player", "Role", "Overall", "Defense", "Attack", "Strength"]]

        print("Team A:")
        print(team_a_summary)
        print(f"Total Strength: {team_a_summary['Strength'].sum():.1f}")
        print("\nTeam B:")
        print(team_b_summary)
        print(f"Total Strength: {team_b_summary['Strength'].sum():.1f}")
        print(f"\nFinal Strength Difference: {most_balanced['difference']:.1f}")

    # Prepare data for plotting
    x = list(range(1, runs + 1))
    team_a_probs = [result[1] for result in results]  # Team A win probabilities
    team_b_probs = [1 - prob for prob in team_a_probs]  # Team B win probabilities

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, team_a_probs, marker="o", label="Team A Win Probability", color="blue")
    plt.plot(x, team_b_probs, marker="o", label="Team B Win Probability", color="green")
    plt.axhline(0.5, color="r", linestyle="--", label="50% Threshold")
    plt.title("Prediction Across Multiple Simulations")
    plt.xlabel("Simulation #")
    plt.ylabel("Win Probability")
    plt.legend()

    # Annotate player lists for the most balanced teams
    if most_balanced["team_a"] is not None and most_balanced["team_b"] is not None:
        team_a_players = ", ".join(most_balanced["team_a"]["Role"].tolist())
        team_b_players = ", ".join(most_balanced["team_b"]["Role"].tolist())
        plt.figtext(0.5, 0.01, f"Most Balanced Teams:\nTeam A: {team_a_players}\nTeam B: {team_b_players}", 
                    wrap=True, horizontalalignment='center', fontsize=10)

    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate balanced soccer teams.")
    parser.add_argument(
        "team_size", 
        type=int, 
        choices=[7, 8], 
        help="Number of players per team (7 or 8)."
    )
    parser.add_argument(
        "players", 
        nargs="*",  # Optional argument, defaults to an empty list
        help="List of Players for the week (space-separated)."
    )
    args = parser.parse_args()

    team_size = args.team_size
    players = args.players
    total_required_players = team_size * 2

    # Load player stats
    player_stats = load_player_stats()
    if player_stats.empty:
        print("Error: No player stats available. Ensure the player stats file is populated.")
        return

    # Handle case where no players are provided
    if not players:
        print("No players provided. Generating random players...")
        for i in range(total_required_players):
            players.append(f"Player_{i + 1}")  # Generate player names like Player_1, Player_2, ...

    # Handle missing players in stats
    player_stats = handle_missing_players(players, player_stats)

    # Generate teams, filling in additional players if needed
    # team_a, team_b = generate_balanced_teams(players, player_stats, total_required_players)

    # Train the model
    model = train_model()

    # Print detailed output
    # print("\nDetailed Output for UI Development:")
    # print(f"Team A Strength: {team_a['Strength'].sum():.1f}")
    # print(f"Team B Strength: {team_b['Strength'].sum():.1f}")
    # print(f"Predicted Outcome: {outcome}")

    # # Save teams to CSV
    # team_output = pd.concat([team_a.assign(Team="Team A"), team_b.assign(Team="Team B")])
    # team_output.to_csv(TEAMS_FILE, index=False)
    # print(f"Teams saved to {TEAMS_FILE}")

    # # Print team compositions
    # print("\nTeam Composition:\n")
    # print("Team A:")
    # for _, row in team_a.iterrows():
    #     print(f"  - {row['Role']}: Overall: {row['Overall']:.1f}, Defense: {row['Defense']}, Attack: {row['Attack']}")

    # print("\nTeam B:")
    # for _, row in team_b.iterrows():
    #     print(f"  - {row['Role']}: Overall: {row['Overall']:.1f}, Defense: {row['Defense']}, Attack: {row['Attack']}")

    # Simulate multiple runs and visualize results
    simulate_runs(players, player_stats, model, runs=100, team_size=team_size)

    # Save updated player stats
    save_player_stats(player_stats)

    # First prediction
    # prediction, outcome = predict_outcome(model, team_a, team_b)

if __name__ == "__main__":
    main()
