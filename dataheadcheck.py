import pandas as pd

# Check WorldCupMatches.csv
matches = pd.read_csv('/home/seed/PROJECT/Data/WorldCupMatches.csv')
print("WorldCupMatches.csv - Head:")
print(matches.head())

# Check WorldCupPlayers.csv
players = pd.read_csv('/home/seed/PROJECT/Data/WorldCupPlayers.csv')
print("\nWorldCupPlayers.csv - Head:")
print(players.head())

# Check WorldCups.csv
world_cups = pd.read_csv('/home/seed/PROJECT/Data/WorldCups.csv')
print("\nWorldCups.csv - Head:")
print(world_cups.head())
