def simulate_runs(players, player_stats, model, runs=10, team_size=7):
    """
    Simulate multiple runs of team balancing and predictions, tracking the most balanced
    team and the run closest to 50-50 win probabilities.
    """
    team_a_wins = 0
    team_b_wins = 0
    total_players = team_size * 2  # Calculate the total number of players needed

    results = []
    top_balanced = []  # List to store the top balanced teams

    # Track the run closest to 50-50 win probabilities
    closest_to_50_50 = {"run": None, "team_a_prob": None, "difference": float("inf")}

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

        # Add the current teams to the list of top balanced teams
        top_balanced.append(
            {
                "difference": difference,
                "team_a": team_a.copy(),
                "team_b": team_b.copy(),
                "run": i + 1,  # Store the run number for annotation
            }
        )

        # Keep only the top 3 most balanced teams
        top_balanced = sorted(top_balanced, key=lambda x: x["difference"])[:3]

        # Track the run closest to 50-50 probabilities
        prob_difference = abs(prediction - 0.5)
        if VERBOSE:
            print(
                f"Run {i+1}: Team A Prob = {prediction:.3f}, Diff from 50% = {prob_difference:.3f}"
            )
        if prob_difference < closest_to_50_50["difference"]:
            closest_to_50_50["difference"] = prob_difference
            closest_to_50_50["run"] = i + 1
            closest_to_50_50["team_a_prob"] = prediction

        # Collect results for analysis and plotting
        results.append((i + 1, prediction, winner, team_a, team_b))

        # Track wins
        if winner == "Team A Wins":
            team_a_wins += 1
        else:
            team_b_wins += 1

    # Debug: Verify closest_to_50_50
    print(
        f"Closest to 50-50 Run: {closest_to_50_50['run']}, Prob Diff: {closest_to_50_50['difference']:.3f}"
    )

    # Print simulation summary
    print(f"\nSimulation Results ({runs} Runs):")
    print(f"Team A Wins: {team_a_wins}")
    print(f"Team B Wins: {team_b_wins}")

    # Display the top 3 most balanced teams
    print("\nTop 3 Most Balanced Team Compositions:")
    for idx, balanced_team in enumerate(top_balanced, start=1):
        print(f"\nBalanced Team {idx}:")
        team_a_summary = balanced_team["team_a"].copy()
        team_b_summary = balanced_team["team_b"].copy()

        # Ensure we use the correct columns
        if "Player" not in team_a_summary.columns:
            team_a_summary["Player"] = team_a_summary.index
        if "Player" not in team_b_summary.columns:
            team_b_summary["Player"] = team_b_summary.index

        # Reorder columns to show 'Player' alongside their attributes
        team_a_summary = team_a_summary[
            ["Player", "Role", "Overall", "Defense", "Attack", "Strength"]
        ]
        team_b_summary = team_b_summary[
            ["Player", "Role", "Overall", "Defense", "Attack", "Strength"]
        ]

        print("Team A:")
        print(team_a_summary)
        print(f"Total Strength: {team_a_summary['Strength'].sum():.1f}")
        print("\nTeam B:")
        print(team_b_summary)
        print(f"Total Strength: {team_b_summary['Strength'].sum():.1f}")
        print(f"Strength Difference: {balanced_team['difference']:.1f}")

    # Let the user select one of the top 3 teams
    selected_team_idx = int(input("\nSelect a team from the top 3 (1, 2, or 3): ")) - 1
    selected_team = top_balanced[selected_team_idx]

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

    # Annotate the run closest to 50-50 probabilities
    if closest_to_50_50["run"] is not None:
        run_idx = (
            closest_to_50_50["run"] - 1
        )  # Convert to 0-based index for accessing lists
        prob_team_a = closest_to_50_50["team_a_prob"]

        # Adjust label position slightly based on the y-value to avoid overlap
        y_offset = 0.02 if prob_team_a > 0.5 else -0.02

        plt.annotate(
            f"Closest to 50-50\nRun {closest_to_50_50['run']}",
            xy=(closest_to_50_50["run"], prob_team_a),
            xytext=(closest_to_50_50["run"], prob_team_a + y_offset),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=10,
            color="black",
        )

    # Annotate team details for the selected team
    team_a_players = ", ".join(selected_team["team_a"]["Player"].tolist())
    team_b_players = ", ".join(selected_team["team_b"]["Player"].tolist())
    plt.figtext(
        0.5,
        0.01,
        f"Selected Balanced Teams:\nTeam A: {team_a_players}\nTeam B: {team_b_players}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )

    plt.show()
