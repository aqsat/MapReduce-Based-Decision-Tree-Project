#!/usr/bin/env python3
import sys

current_team = None
total_scored = 0
total_conceded = 0
match_count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        team, scored, conceded, count = line.split("\t")
        scored = int(scored)
        conceded = int(conceded)
        count = int(count)
    except (ValueError, IndexError):
        print(f"Skipping invalid input: {line}", file=sys.stderr)
        continue
    if current_team == team:
        total_scored += scored
        total_conceded += conceded
        match_count += count
    else:
        if current_team is not None:
            avg_scored = total_scored / match_count if match_count > 0 else 0
            avg_conceded = total_conceded / match_count if match_count > 0 else 0
            print(f"{current_team}\t{avg_scored:.2f}\t{avg_conceded:.2f}\t{match_count}")
        current_team = team
        total_scored = scored
        total_conceded = conceded
        match_count = count
if current_team is not None:
    avg_scored = total_scored / match_count if match_count > 0 else 0
    avg_conceded = total_conceded / match_count if match_count > 0 else 0
    print(f"{current_team}\t{avg_scored:.2f}\t{avg_conceded:.2f}\t{match_count}")