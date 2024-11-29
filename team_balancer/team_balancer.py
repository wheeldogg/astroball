import os

import pandas as pd
import random
import argparse

import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# File paths
PLAYER_STATS_FILE = "data/player_stats.csv"
HISTORICAL_RESULTS_FILE = "data/historical_results.csv"
TEAMS_FILE = "data/teams.csv"

# Globals
VERBOSE = False
FACTORS_TO_USE = ["Overall", "Defense", "Attack", "Stamina", "Versatility", "Role"]
ANNOTATE_PLAYERS_ON_PLOT = False


# Function to load or create player stats
def load_player_stats():
    if os.path.exists(PLAYER_STATS_FILE):
        return pd.read_csv(PLAYER_STATS_FILE)
    else:
        columns = FACTORS_TO_USE
        return pd.DataFrame(columns=columns)


def calculate_strength(row):
    """
    Calculate player strength using weighted attributes.
    Adjust the weights based on importance and the data ranges.
    """
    return (
        row["Overall"] * 0.9  # Overall is a significant contributor
        + row["Defense"] * 0.25  # Defense is important but secondary
        + row["Attack"] * 0.25  # Attack contributes similarly to Defense
        + row["Stamina"] * 0.25  # Stamina plays a moderate role
        + row["Versatility"] * 0.25  # Versatility is a minor factor
    )


def generate_demo_players(player_count=14):
    """
    Generate a demo dataset of players with random stats.
    """
    roles = ["Defender", "Midfielder", "Striker", "Goalkeeper"]
    demo_data = {
        "Player": [f"Player_{i+1}" for i in range(player_count)],
        "Role": [random.choice(roles) for _ in range(player_count)],
        "Overall": np.random.randint(60, 100, player_count),
        "Defense": np.random.randint(30, 90, player_count),
        "Attack": np.random.randint(30, 90, player_count),
        "Stamina": np.random.randint(50, 100, player_count),
        "Versatility": np.random.randint(40, 80, player_count),
    }
    return pd.DataFrame(demo_data)


def balance_teams(player_data, strength_threshold=10, protected_players=None):
    """
    Balance two teams based on their strength while ensuring that protected players
    are not swapped. Players are swapped between teams to minimize the strength difference.

    Parameters:
        player_data (pd.DataFrame): DataFrame containing player information with 'Strength' and 'Player' columns.
        strength_threshold (int): Maximum acceptable strength difference between the two teams.
        protected_players (list): List of player names who cannot be swapped.

    Returns:
        team_a (pd.DataFrame), team_b (pd.DataFrame): Balanced teams.
    """
    if protected_players is None:
        protected_players = []

    # Shuffle the player data to introduce randomness
    shuffled_players = player_data.sample(
        frac=1, random_state=random.randint(1, 10000)
    ).reset_index(drop=True)

    # Split players into two teams
    team_a = []
    team_b = []
    for _, row in shuffled_players.iterrows():
        if len(team_a) <= len(team_b):
            team_a.append(row)
        else:
            team_b.append(row)

    # Convert lists to DataFrames
    team_a = pd.DataFrame(team_a)
    team_b = pd.DataFrame(team_b)

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
            player_to_swap_out = team_a_candidates.loc[
                team_a_candidates["Strength"].idxmin()
            ]
            player_to_swap_in = team_b_candidates.loc[
                team_b_candidates["Strength"].idxmax()
            ]
        else:
            # Team B is stronger
            team_b_candidates = team_b[~team_b["Player"].isin(protected_players)]
            team_a_candidates = team_a[~team_a["Player"].isin(protected_players)]
            if team_b_candidates.empty or team_a_candidates.empty:
                break  # No candidates available for swapping
            player_to_swap_out = team_b_candidates.loc[
                team_b_candidates["Strength"].idxmin()
            ]
            player_to_swap_in = team_a_candidates.loc[
                team_a_candidates["Strength"].idxmax()
            ]

        # Perform the swap
        if strength_a > strength_b:
            # Swap weakest in Team A with strongest in Team B
            team_a = team_a.drop(player_to_swap_out.name)
            team_b = team_b.drop(player_to_swap_in.name)
            team_a = pd.concat(
                [team_a, player_to_swap_in.to_frame().T], ignore_index=True
            )
            team_b = pd.concat(
                [team_b, player_to_swap_out.to_frame().T], ignore_index=True
            )
        else:
            # Swap weakest in Team B with strongest in Team A
            team_b = team_b.drop(player_to_swap_out.name)
            team_a = team_a.drop(player_to_swap_in.name)
            team_b = pd.concat(
                [team_b, player_to_swap_in.to_frame().T], ignore_index=True
            )
            team_a = pd.concat(
                [team_a, player_to_swap_out.to_frame().T], ignore_index=True
            )

    # Final check
    final_difference = abs(team_a["Strength"].sum() - team_b["Strength"].sum())
    if VERBOSE:
        print(f"Final strength difference after balancing: {final_difference:.1f}")
    return team_a, team_b


def generate_balanced_teams(players, player_stats, total_players, team_size):
    """
    Generate balanced teams while ensuring that the provided players are always included
    and that roles (e.g., Attackers, Defenders) are balanced.
    """
    # Create the final player dataset
    player_data = player_stats[player_stats["Player"].isin(players)].copy()

    # Identify missing players
    missing_players = [
        player for player in players if player not in player_data["Player"].values
    ]
    if missing_players:
        raise ValueError(
            f"The following players are not found in the dataframe: {missing_players}"
        )

    # Check if we have the correct number of players
    if len(player_data.index) != team_size * 2:
        raise ValueError(
            f"Incorrect number of players in the dataframe. Expected {team_size * 2}, but got {len(player_data.index)}. "
            f"Missing players: {missing_players}"
        )

    # Balance teams by role
    team_a, team_b = [], []
    roles = player_data["Role"].unique()
    for role in roles:
        players_by_role = player_data[player_data["Role"] == role].sample(
            frac=1, random_state=random.randint(1, 10000)
        )  # Shuffle players in this role
        for _, row in players_by_role.iterrows():
            if len(team_a) <= len(team_b):
                team_a.append(row)
            else:
                team_b.append(row)

    # Convert lists to DataFrames
    team_a = pd.DataFrame(team_a)
    team_b = pd.DataFrame(team_b)

    # Calculate strength and balance teams further
    team_a["Strength"] = team_a.apply(calculate_strength, axis=1)
    team_b["Strength"] = team_b.apply(calculate_strength, axis=1)
    team_a, team_b = balance_teams(pd.concat([team_a, team_b]), strength_threshold=10)

    return team_a, team_b


# Function to train the prediction model
def train_model():
    if os.path.exists(HISTORICAL_RESULTS_FILE):
        historical_data = pd.read_csv(HISTORICAL_RESULTS_FILE)
        X = historical_data[
            ["Team1_Strength", "Team2_Strength", "Team1_Defense", "Team2_Attack"]
        ]
        y = historical_data["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(
            f"Model trained. Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}"
        )
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

    features = pd.DataFrame(
        [[team1_strength, team2_strength, team1_defense, team2_attack]],
        columns=["Team1_Strength", "Team2_Strength", "Team1_Defense", "Team2_Attack"],
    )
    prediction = model.predict(features)[0]
    winner = "Team A Wins" if prediction > 0.5 else "Team B Wins"
    return prediction, winner


def simulate_runs(players, player_stats, model, runs=10, team_size=7):
    """
    Simulate multiple runs of team balancing and predictions, tracking the most balanced
    team and plotting team strengths and their difference over runs.
    """
    team_a_wins = 0
    team_b_wins = 0
    total_players = team_size * 2  # Calculate the total number of players needed

    results = []
    strengths_over_runs = []  # Track Team A, Team B strengths, and their difference

    # Track the top 3 permutations with the minimal strength difference
    top_balanced = []

    for i in range(runs):
        # Generate balanced teams for each run
        team_a, team_b = generate_balanced_teams(
            players, player_stats, total_players, team_size
        )

        # Predict outcome
        prediction, winner = predict_outcome(model, team_a, team_b)
        if prediction is None:
            break

        # Calculate team strengths and difference
        strength_a = team_a["Strength"].sum()
        strength_b = team_b["Strength"].sum()
        difference = abs(strength_a - strength_b)

        if VERBOSE:
            print(
                f"Run {i+1}: Team A Strength = {strength_a:.1f}, Team B Strength = {strength_b:.1f}, Difference = {difference:.1f}"
            )

        # Track the top 3 most balanced runs
        top_balanced.append(
            {
                "run": i + 1,
                "difference": difference,
                "team_a_strength": strength_a,
                "team_b_strength": strength_b,
                "team_a": team_a.copy(),
                "team_b": team_b.copy(),
            }
        )
        top_balanced = sorted(top_balanced, key=lambda x: x["difference"])[:3]

        # Record strengths for plotting
        strengths_over_runs.append((i + 1, strength_a, strength_b, difference))

        # Collect results for analysis
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

    # Display the top 3 most balanced teams
    print("\nTop 3 Most Balanced Team Compositions:")
    for idx, balanced in enumerate(top_balanced, start=1):
        print(f"\nBalanced Team {idx}: Run {balanced['run']}")
        print(f"Difference: {balanced['difference']:.1f}")

        # Format Team A Players and Roles
        team_a_players_roles = [
            f"{row['Player']} ({row['Role']})"
            for _, row in balanced["team_a"].iterrows()
        ]
        team_a_names = balanced["team_a"]["Player"].tolist()  # Names-only list
        team_a_role_counts = balanced["team_a"]["Role"].value_counts().to_dict()
        print("Team A Players (with roles):", ", ".join(team_a_players_roles))
        print("Team A Players (names only):", ", ".join(team_a_names))
        print(
            "Team A Role Counts:",
            ", ".join(
                [f"{role}: {count}" for role, count in team_a_role_counts.items()]
            ),
        )

        # Format Team B Players and Roles
        team_b_players_roles = [
            f"{row['Player']} ({row['Role']})"
            for _, row in balanced["team_b"].iterrows()
        ]
        team_b_names = balanced["team_b"]["Player"].tolist()  # Names-only list
        team_b_role_counts = balanced["team_b"]["Role"].value_counts().to_dict()
        print("Team B Players (with roles):", ", ".join(team_b_players_roles))
        print("Team B Players (names only):", ", ".join(team_b_names))
        print(
            "Team B Role Counts:",
            ", ".join(
                [f"{role}: {count}" for role, count in team_b_role_counts.items()]
            ),
        )

    # Prompt the user to select one of the top 3 teams
    selected_team_idx = int(input("\nSelect a team from the top 3 (1, 2, or 3): ")) - 1
    selected_team = top_balanced[selected_team_idx]

    # Plot Team Strengths and Differences
    x = [entry[0] for entry in strengths_over_runs]  # Run numbers
    team_a_strengths = [entry[1] for entry in strengths_over_runs]
    team_b_strengths = [entry[2] for entry in strengths_over_runs]
    strength_differences = [entry[3] for entry in strengths_over_runs]

    plt.figure(figsize=(12, 6))
    plt.plot(x, team_a_strengths, marker="o", label="Team A Strength", color="blue")
    plt.plot(x, team_b_strengths, marker="o", label="Team B Strength", color="green")
    plt.plot(
        x,
        strength_differences,
        marker="o",
        label="Strength Difference",
        color="red",
        linestyle="--",
    )

    # Annotate the selected run
    selected_run = selected_team["run"]
    selected_diff = selected_team["difference"]
    team_a_players = ", ".join(selected_team["team_a"]["Player"].tolist())
    team_b_players = ", ".join(selected_team["team_b"]["Player"].tolist())

    # Get the midpoint of the Y-axis for vertical centering
    y_mid = (min(strength_differences) + max(strength_differences)) / 2

    # Annotate the selected run with team details, positioned halfway up the plot
    if ANNOTATE_PLAYERS_ON_PLOT:
        plt.annotate(
            f"Selected Run\nRun {selected_run}\nDiff: {selected_diff:.1f}\n"
            f"Team A: {team_a_players}\nTeam B: {team_b_players}",
            xy=(selected_run, selected_diff),  # Point to the relevant run
            xytext=(
                selected_run - 5,
                y_mid,
            ),  # Shift annotation to the middle of the plot
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
            horizontalalignment="center",  # Center-align text
            verticalalignment="center",  # Vertically center the text box
        )
    else:
        plt.annotate(
            f"Selected Run\nRun {selected_run}\nDiff: {selected_diff:.1f}",
            xy=(selected_run, selected_diff),
            xytext=(selected_run + 1, selected_diff + 2),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=10,
            color="black",
        )

    plt.title("Team Strengths and Differences Across Simulations")
    plt.xlabel("Simulation Run")
    plt.ylabel("Strength")
    plt.legend()
    plt.grid()
    plt.show()


def check_team_size(args=None):
    team_size = args.team_size
    players = args.players
    total_required_players = team_size * 2
    if len(players) != total_required_players:
        raise ValueError(
            f"Total amount of players not met in the input. total_required_players os {total_required_players}. Players {len(players)}"
        )
    return players, total_required_players, team_size


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate balanced soccer teams.")
    parser.add_argument(
        "team_size",
        type=int,
        choices=[7, 8, 9],
        help="Number of players per team (7, 8 or 9).",
    )
    parser.add_argument(
        "players",
        nargs="*",  # Optional argument, defaults to an empty list
        help="List of Players for the week (space-separated).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo mode to generate 14 random players.",
    )
    args = parser.parse_args()

    if args.demo:
        print("Demo mode enabled. Generating random player data...")
        player_stats = generate_demo_players(player_count=args.team_size * 2)
        players = player_stats["Player"].tolist()
        total_required_players = args.team_size * 2
        team_size = args.team_size
    else:
        players, total_required_players, team_size = check_team_size(args=args)
        print(f"Players Required: {players}")
        print(f"Total Required Players: {total_required_players}")

        # Load player stats
        player_stats = load_player_stats()
        if player_stats.empty:
            raise ValueError(
                "Error: No player stats available. Ensure the player stats file is populated."
            )

    # Simulate multiple runs and visualize results
    model = train_model()
    simulate_runs(players, player_stats, model, runs=100, team_size=team_size)
    print("\n\nFinished simulate runs\n\n")


if __name__ == "__main__":
    main()
