#!/usr/bin/env python3
import sys
import csv
import re

# Open a debug log file
with open("mapper_debug.log", "w") as debug_log:
    print("DEBUG: Mapper started", file=debug_log, flush=True)

    invalid_count = 0
    max_invalid = 10

    for line in sys.stdin:
        line = line.rstrip()
        if not line:
            print("DEBUG: Skipping empty line", file=debug_log, flush=True)
            continue

        print(f"DEBUG: Processing line: {line}", file=debug_log, flush=True)

        try:
            reader = csv.reader([line], skipinitialspace=True)
            fields = next(reader)
            print(f"DEBUG: Parsed fields: {fields}", file=debug_log, flush=True)
        except Exception as e:
            invalid_count += 1
            if invalid_count <= max_invalid:
                print(f"DEBUG: Invalid line {invalid_count}: {line}, Error: {e}", file=debug_log, flush=True)
            continue

        if fields[0].lower().strip() == "year" or fields[5].lower().strip() == "home team name":
            print(f"DEBUG: Skipping header: {line}", file=debug_log, flush=True)
            continue

        if len(fields) < 9:
            invalid_count += 1
            if invalid_count <= max_invalid:
                print(f"DEBUG: Skipping line with insufficient fields: {line}", file=debug_log, flush=True)
            continue

        home_team = fields[5].strip().lower()
        home_goals = fields[6].strip()
        away_goals = fields[7].strip()
        away_team = fields[8].strip().lower()

        home_team = re.sub(r'[^a-zA-Z\s]', '', home_team).strip()
        away_team = re.sub(r'[^a-zA-Z\s]', '', away_team).strip()

        if not home_team or not away_team:
            invalid_count += 1
            if invalid_count <= max_invalid:
                print(f"DEBUG: Skipping line with empty team names: {line}", file=debug_log, flush=True)
            continue

        try:
            home_goals = int(home_goals)
            away_goals = int(away_goals)
        except ValueError:
            invalid_count += 1
            if invalid_count <= max_invalid:
                print(f"DEBUG: Skipping line with non-integer goals: {line}", file=debug_log, flush=True)
            continue

        print(f"{home_team}\t{home_goals}")
        print(f"{away_team}\t{away_goals}")
        print(f"DEBUG: Emitted: {home_team}\t{home_goals}, {away_team}\t{away_goals}", file=debug_log, flush=True)

        if "brazil" in home_team or "brazil" in away_team:
            print(f"DEBUG: Brazil match: {home_team} {home_goals}, {away_team} {away_goals}", file=debug_log, flush=True)
