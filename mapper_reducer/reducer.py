#!/usr/bin/env python3
import sys

current_team = None
current_goals = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        team, goals = line.split("\t")
        goals = int(goals)
    except (ValueError, IndexError):
        print(f"Skipping invalid input: {line}", file=sys.stderr)
        continue
    
    if current_team == team:
        current_goals += goals
    else:
        if current_team is not None:
            print(f"{current_team}\t{current_goals}")
        current_team = team
        current_goals = goals

# Output the last team
if current_team is not None:
    print(f"{current_team}\t{current_goals}")