# Save as check_matches.py
import csv

valid_matches = 0
with open('/home/seed/PROJECT/Data/WorldCupMatches_clean.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 9 or row[0].lower().strip() == 'year':
            continue
        try:
            home_team = row[5].strip()
            away_team = row[8].strip()
            home_goals = int(row[6].strip())
            away_goals = int(row[7].strip())
            if home_team and away_team:
                valid_matches += 1
        except (ValueError, IndexError):
            continue
print(f"Valid matches: {valid_matches}")