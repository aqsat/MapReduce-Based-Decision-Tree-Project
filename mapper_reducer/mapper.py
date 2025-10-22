
#/usr/bin/env python3
import sys
import csv

# Track invalid lines for debugging
invalid_count = 0
max_invalid = 10

for line in sys.stdin:
    line = line.rstrip()  # Remove trailing newlines
    if not line:
        print("DEBUG: Skipping empty line", file=sys.stderr)
        continue

    # Parse line as CSV
    try:
        reader = csv.reader([line], skipinitialspace=True)
        fields = next(reader)
        print(f"DEBUG: Parsed fields: {fields}", file=sys.stderr)
    except Exception as e:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Invalid line {invalid_count}: {line}, Error: {e}", file=sys.stderr)
        continue

    # Skip header row
    if fields[0].lower().strip() == "year" or fields[5].lower().strip() == "home team name":
        print(f"DEBUG: Skipping header: {line}", file=sys.stderr)
        continue

    # Expect at least 9 fields
    if len(fields) < 9:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with insufficient fields: {line}", file=sys.stderr)
        continue

    # Extract and clean fields
    home_team = fields[5].strip().lower()  # Normalize team names
    home_goals = fields[6].strip()
    away_goals = fields[7].strip()
    away_team = fields[8].strip().lower()   # Normalize team names

    # Remove any stray quotes or prefixes (e.g., 'rn">')
    home_team = home_team.replace('"', '').replace('rn>', '')
    away_team = away_team.replace('"', '').replace('rn>', '')

    # Validate team names
    if not home_team or not away_team:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with empty team names: {line}", file=sys.stderr)
        continue

    # Validate goals
    try:
        home_goals = int(home_goals)
        away_goals = int(away_goals)
    except ValueError:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with non-integer goals: {line}", file=sys.stderr)
        continue

    # Emit key-value pairs
    print(f"{home_team}\t{home_goals}")
    print(f"{away_team}\t{away_goals}")
    print(f"DEBUG: Emitted: {home_team}\t{home_goals}, {away_team}\t{away_goals}", file=sys.stderr)

    # Log Brazil-specific emissions
    if home_team == "brazil" or away_team == "brazil":
        print(f"DEBUG: Brazil match: {home_team} {home_goals}, {away_team} {away_goals}", file=sys.stderr)