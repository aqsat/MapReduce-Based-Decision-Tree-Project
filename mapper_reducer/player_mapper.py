#!/usr/bin/env python3
import sys
import csv
import re

# Read input as CSV to handle commas in fields properly
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    # Parse the line as CSV
    reader = csv.reader([line], skipinitialspace=True)
    fields = next(reader)
    if fields[0] == "RoundID":  # Skip header
        continue
    try:
        team = fields[2]
        match_id = fields[1]
        player_name = fields[6]
        line_up = fields[4]
        # Handle position field (index 7) carefully
        position = fields[7] if len(fields) > 7 and fields[7] else ""
        # Handle event field (index 8)
        event = fields[8] if len(fields) > 8 and fields[8] else ""

        # Emit for counting unique matches
        print(f"{team}\tmatch:{match_id}")

        # Emit for goals and players who scored
        if "G" in event:
            goals = re.findall(r"G\d+'", event)
            for _ in goals:
                print(f"{team}\tgoal:{player_name}")

        # Emit for starting players
        if line_up == "S":
            print(f"{team}\tstarter:{match_id}")

        # Emit for substitutions (I for in, O for out)
        if "I" in event or "O" in event:
            print(f"{team}\tsubstitution:{match_id}")

        # Emit for yellow cards
        if "Y" in event:
            print(f"{team}\tyellow:{match_id}:{player_name}")

        # Emit for red cards
        if "R" in event:
            print(f"{team}\tred:{match_id}:{player_name}")

        # Emit for captains (check if position contains 'C')
        if position in ("C", "GKC"):
            print(f"{team}\tcaptain:{player_name}")

        # Emit for goalkeepers (check if position contains 'GK')
        if position in ("GK", "GKC"):
            print(f"{team}\tgoalkeeper:{player_name}")

    except (IndexError, ValueError) as e:
        # Log error for debugging (optional)
        # sys.stderr.write(f"Error processing line: {line}, Error: {e}\n")
        continue