#!/usr/bin/env python3
import sys

current_team = None
wins = 0
losses = 0
draws = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        team, outcome = line.split("\t")
    except (ValueError, IndexError):
        print(f"Skipping invalid input: {line}", file=sys.stderr)
        continue
    if current_team == team:
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "draw":
            draws += 1
    else:
        if current_team is not None:
            print(f"{current_team}\t{wins}\t{losses}\t{draws}")
        current_team = team
        wins = 1 if outcome == "win" else 0
        losses = 1 if outcome == "loss" else 0
        draws = 1 if outcome == "draw" else 0
if current_team is not None:
    print(f"{current_team}\t{wins}\t{losses}\t{draws}")