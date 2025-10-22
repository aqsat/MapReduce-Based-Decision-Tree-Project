#!/usr/bin/env python3
import sys

current_team = None
current_count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        team, count = line.split("\t")
        count = int(count)
    except (ValueError, IndexError):
        print(f"Skipping invalid input: {line}", file=sys.stderr)
        continue
    if current_team == team:
        current_count += count
    else:
        if current_team is not None:
            print(f"{current_team}\t{current_count}")
        current_team = team
        current_count = count
if current_team is not None:
    print(f"{current_team}\t{current_count}")