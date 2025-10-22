import pandas as pd

# Load the WorldCupMatches.csv file to inspect the column names
matches = pd.read_csv('/home/seed/PROJECT/Data/WorldCupMatches.csv')

# Print the column names to find the correct ones for home and away teams
print(matches.columns)
