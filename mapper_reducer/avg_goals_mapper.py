#!/usr/bin/env python3
import sys
import csv

invalid_count = 0
max_invalid = 10

for line in sys.stdin:
    line = line.rstrip()
    if not line:
        continue
    try:
        reader = csv.reader([line], skipinitialspace=True)
        fields = next(reader)
    except Exception as e:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Invalid line {invalid_count}: {line}, Error: {e}", file=sys.stderr)
        continue
    if fields[0].lower().strip() == "year":
        continue
    if len(fields) < 9:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with insufficient fields: {line}", file=sys.stderr)
        continue
    home_team = fields[5].strip().lower().replace('"', '').replace('rn>', '')
    away_team = fields[8].strip().lower().replace('"', '').replace('rn>', '')
    try:
        home_goals = int(fields[6].strip())
        away_goals = int(fields[7].strip())
    except ValueError:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with non-integer goals: {line}", file=sys.stderr)
        continue
    if not home_team or not away_team:
        invalid_count += 1
        if invalid_count <= max_invalid:
            print(f"DEBUG: Skipping line with empty team names: {line}", file=sys.stderr)
        continue
    # Emit: team, scored, conceded, match_count
    print(f"{home_team}\t{home_goals}\t{away_goals}\t1")
    print(f"{away_team}\t{away_goals}\t{home_goals}\t1")