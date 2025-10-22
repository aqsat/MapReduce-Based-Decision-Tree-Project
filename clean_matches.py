# Save as clean_matches.py
import csv

input_file = '/home/seed/PROJECT/Data/WorldCupMatches.csv'
output_file = '/home/seed/PROJECT/Data/WorldCupMatches_clean.csv'

with open(input_file, 'r') as f, open(output_file, 'w', newline='') as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    header = next(reader)  # Write header
    writer.writerow(header)
    for row in reader:
        if len(row) < 9:
            continue
        try:
            home_team = row[5].strip()
            away_team = row[8].strip()
            home_goals = int(row[6].strip())
            away_goals = int(row[7].strip())
            if home_team and away_team:
                writer.writerow(row)
        except (ValueError, IndexError):
            continue