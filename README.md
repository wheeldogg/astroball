# astroball

Astroball 7 and 8 a side soccer team generator

Setup Instructions for New Users
Clone the Repository:

```bash
Copy code
git clone https://github.com/yourusername/team-balancer.git
cd team-balancer
Install Dependencies: Ensure you have Poetry installed. Then:
```

```bash
Copy code
poetry install
Run the Script: Provide the weekly player names and run the script:
```

```bash
Copy code
poetry run python team_balancer.py
Check Outputs:
```

Updated data/teams.csv with weekly team assignments.
Updated data/player_stats.csv if new players were added or stats were learned.


## Input data

Create a csv called `player_stats.csv`

```csv
Player,Defense,Attack,Stamina,Versatility,Speed,Passing,Positioning,Aggression,Wins,Losses,Goals,Assists,MVP,Attendance,Experience,Preferred Role,Tactical Notes
Player1,4,3,5,5,,,,,6,4,,,,3,1,,,Defender,
Player2,3,4,4,2,,,,,5,5,,,,2,1,,,Midfielder,
Player3,1,5,3,3,,,,,7,7,,,,0,1,,,Forward,
Player4,4,3,4,4,,,,,6,6,,,,1,0,,,Defender,
Player5,3,4,5,3,,,,,8,5,,,,2,1,,,Midfielder,
Player6,2,5,3,4,,,,,4,7,,,,1,0,,,Forward,
Player7,4,2,5,4,,,,,5,5,,,,3,1,,,Defender,
Player8,3,3,4,3,,,,,6,6,,,,0,0,,,Midfielder,
Player9,5,2,4,5,,,,,7,4,,,,2,1,,,Defender,
Player10,2,5,3,3,,,,,4,8,,,,0,0,,,Forward,
Player11,3,4,4,4,,,,,6,5,,,,1,1,,,Midfielder,
Player12,4,3,5,3,,,,,5,6,,,,3,1,,,Defender,
Player13,1,5,3,2,,,,,7,7,,,,0,1,,,Forward,
Player14,4,2,5,5,,,,,6,4,,,,2,2,,,Defender,
```

How to Use This CSV
Important Attributes (Populated):
Defense, Attack, Stamina, Versatility, Wins, MVP, Attendance, Preferred Role.
Less Important Attributes (Set to null):
Speed, Passing, Positioning, Aggression, Goals, Assists, Experience, Tactical Notes.


## Example to run

`poetry run python team_balancer.py Player1 Player2 Player3`


## Output

Model trained. Mean Squared Error: 0.05
Teams saved to teams.csv

Detailed Output for UI Development:
Team A Strength: 55.8
Team B Strength: 54.2
Team A Defense: 15
Team B Attack: 20
Predicted Outcome: Team A Wins

Simulation Results (10 Runs):
Team A Wins: 5
Team B Wins: 5