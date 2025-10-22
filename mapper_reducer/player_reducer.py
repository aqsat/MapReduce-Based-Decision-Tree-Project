#!/usr/bin/env python3
import sys

current_team = None
matches = set()  # Track unique matches
goals = 0  # Total goals
players_scored = set()  # Unique players who scored
starters_per_match = {}  # Number of starters per match
substitutions = set()  # Unique matches with substitutions
yellow_cards = set()  # Unique yellow card events
red_cards = set()  # Unique red card events
captains = set()  # Unique captains
goalkeepers = set()  # Unique goalkeepers

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    team, value = line.split('\t')

    if team != current_team:
        if current_team:
            # Compute average starters per match
            total_starters = sum(starters_per_match.values())
            num_matches = len(matches)
            avg_starters = total_starters / num_matches if num_matches > 0 else 0
            # Compute average goals per match
            avg_goals = goals / num_matches if num_matches > 0 else 0
            # Output all features for the previous team
            print(f"{current_team}\t{num_matches}\t{goals}\t{len(players_scored)}\t{avg_starters:.2f}\t"
                  f"{len(substitutions)}\t{len(yellow_cards)}\t{len(red_cards)}\t{len(captains)}\t"
                  f"{len(goalkeepers)}\t{avg_goals:.2f}")
        # Reset for the new team
        current_team = team
        matches.clear()
        goals = 0
        players_scored.clear()
        starters_per_match.clear()
        substitutions.clear()
        yellow_cards.clear()
        red_cards.clear()
        captains.clear()
        goalkeepers.clear()

    # Process the value based on its prefix
    if value.startswith("match:"):
        match_id = value.split(":", 1)[1]
        matches.add(match_id)
    elif value.startswith("goal:"):
        player_name = value.split(":", 1)[1]
        players_scored.add(player_name)
        goals += 1
    elif value.startswith("starter:"):
        match_id = value.split(":", 1)[1]
        starters_per_match[match_id] = starters_per_match.get(match_id, 0) + 1
    elif value.startswith("substitution:"):
        match_id = value.split(":", 1)[1]
        substitutions.add(match_id)
    elif value.startswith("yellow:"):
        yellow_key = value.split(":", 1)[1]
        yellow_cards.add(yellow_key)
    elif value.startswith("red:"):
        red_key = value.split(":", 1)[1]
        red_cards.add(red_key)
    elif value.startswith("captain:"):
        captain_name = value.split(":", 1)[1]
        captains.add(captain_name)
    elif value.startswith("goalkeeper:"):
        gk_name = value.split(":", 1)[1]
        goalkeepers.add(gk_name)

# Output the last team
if current_team:
    total_starters = sum(starters_per_match.values())
    num_matches = len(matches)
    avg_starters = total_starters / num_matches if num_matches > 0 else 0
    avg_goals = goals / num_matches if num_matches > 0 else 0
    print(f"{current_team}\t{num_matches}\t{goals}\t{len(players_scored)}\t{avg_starters:.2f}\t"
          f"{len(substitutions)}\t{len(yellow_cards)}\t{len(red_cards)}\t{len(captains)}\t"
          f"{len(goalkeepers)}\t{avg_goals:.2f}")