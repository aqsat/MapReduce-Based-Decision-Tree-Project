import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load MapReduce output files
player_data = pd.read_csv('/home/seed/PROJECT/output/player_goals_output.txt', sep='\t', 
                          names=['team_initials', 'num_matches', 'total_goals', 'num_players_scored', 
                                 'avg_starters_per_match', 'num_substitutions', 'num_yellow_cards', 
                                 'num_red_cards', 'num_unique_captains', 'num_unique_goalkeepers', 
                                 'avg_goals_per_match'])
team_matches = pd.read_csv('/home/seed/PROJECT/output/team_matches.txt', sep='\t', names=['team', 'matches_played'])
team_outcomes = pd.read_csv('/home/seed/PROJECT/output/team_outcomes.txt', sep='\t', 
                            names=['team', 'wins', 'draws', 'losses'])
team_avg_goals = pd.read_csv('/home/seed/PROJECT/output/team_avg_goals.txt', sep='\t', 
                             names=['team', 'avg_goals_scored', 'avg_goals_conceded', 'matches_played'])

# Step 2: Map team initials to full names
team_mapping = {
    'ALG': 'algeria', 'ANG': 'angola', 'ARG': 'argentina', 'AUS': 'australia', 'AUT': 'austria',
    'BEL': 'belgium', 'BIH': 'bosnia and herzegovina', 'BOL': 'bolivia', 'BRA': 'brazil', 'BUL': 'bulgaria',
    'CAN': 'canada', 'CHI': 'chile', 'CHN': 'china pr', 'CIV': 'côte d\'ivoire', 'CMR': 'cameroon',
    'COL': 'colombia', 'CRC': 'costa rica', 'CRO': 'croatia', 'CUB': 'cuba', 'CZE': 'czech republic',
    'DEN': 'denmark', 'ECU': 'ecuador', 'EGY': 'egypt', 'ENG': 'england', 'ESP': 'spain',
    'FRA': 'france', 'FRG': 'germany', 'GDR': 'germany', 'GER': 'germany', 'GHA': 'ghana',
    'GRE': 'greece', 'HAI': 'haiti', 'HON': 'honduras', 'HUN': 'hungary', 'INH': 'indonesia',
    'IRL': 'republic of ireland', 'IRN': 'iran', 'IRQ': 'iraq', 'ISR': 'israel', 'ITA': 'italy',
    'JAM': 'jamaica', 'JPN': 'japan', 'KOR': 'korea republic', 'KSA': 'saudi arabia', 'KUW': 'kuwait',
    'MAR': 'morocco', 'MEX': 'mexico', 'NED': 'netherlands', 'NGA': 'nigeria', 'NIR': 'northern ireland',
    'NOR': 'norway', 'NZL': 'new zealand', 'PAR': 'paraguay', 'PER': 'peru', 'POL': 'poland',
    'POR': 'portugal', 'PRK': 'korea dpr', 'ROU': 'romania', 'RSA': 'south africa', 'RUS': 'russia',
    'SCG': 'serbia', 'SCO': 'scotland', 'SEN': 'senegal', 'SLV': 'el salvador', 'SRB': 'serbia',
    'SUI': 'switzerland', 'SVK': 'slovakia', 'SVN': 'slovenia', 'SWE': 'sweden', 'TCH': 'czechoslovakia',
    'TOG': 'togo', 'TRI': 'trinidad and tobago', 'TUN': 'tunisia', 'TUR': 'turkey', 'UAE': 'united arab emirates',
    'UKR': 'ukraine', 'URS': 'soviet union', 'URU': 'uruguay', 'USA': 'united states', 'WAL': 'wales',
    'YUG': 'yugoslavia', 'ZAI': 'zaire'
}
player_data['team'] = player_data['team_initials'].map(team_mapping).fillna(player_data['team_initials'])

# Step 3: Merge datasets
data = pd.merge(player_data, team_matches, on='team', how='left')
data = pd.merge(data, team_outcomes, on='team', how='left')
data = pd.merge(data, team_avg_goals[['team', 'avg_goals_scored', 'avg_goals_conceded']], on='team', how='left')

# Step 4: Debug merges
print("Missing values after merges:")
print(data[['wins', 'draws', 'losses', 'avg_goals_scored', 'avg_goals_conceded']].isna().sum())
# Debug team mismatches
print("Teams in player_data:", sorted(player_data['team'].unique()))
print("Teams in team_matches:", sorted(team_matches['team'].unique()))
print("Teams in team_outcomes:", sorted(team_outcomes['team'].unique()))
print("Teams in team_avg_goals:", sorted(team_avg_goals['team'].unique()))

# Step 5: Feature Engineering
# Region mapping
region_mapping = {
    'ALG': 'Africa', 'ANG': 'Africa', 'ARG': 'South America', 'AUS': 'Asia', 'AUT': 'Europe',
    'BEL': 'Europe', 'BIH': 'Europe', 'BOL': 'South America', 'BRA': 'South America', 'BUL': 'Europe',
    'CAN': 'North America', 'CHI': 'South America', 'CHN': 'Asia', 'CIV': 'Africa', 'CMR': 'Africa',
    'COL': 'South America', 'CRC': 'North America', 'CRO': 'Europe', 'CUB': 'North America', 'CZE': 'Europe',
    'DEN': 'Europe', 'ECU': 'South America', 'EGY': 'Africa', 'ENG': 'Europe', 'ESP': 'Europe',
    'FRA': 'Europe', 'FRG': 'Europe', 'GDR': 'Europe', 'GER': 'Europe', 'GHA': 'Africa',
    'GRE': 'Europe', 'HAI': 'North America', 'HON': 'North America', 'HUN': 'Europe', 'INH': 'Asia',
    'IRL': 'Europe', 'IRN': 'Asia', 'IRQ': 'Asia', 'ISR': 'Asia', 'ITA': 'Europe',
    'JAM': 'North America', 'JPN': 'Asia', 'KOR': 'Asia', 'KSA': 'Asia', 'KUW': 'Asia',
    'MAR': 'Africa', 'MEX': 'North America', 'NED': 'Europe', 'NGA': 'Africa', 'NIR': 'Europe',
    'NOR': 'Europe', 'NZL': 'Oceania', 'PAR': 'South America', 'PER': 'South America', 'POL': 'Europe',
    'POR': 'Europe', 'PRK': 'Asia', 'ROU': 'Europe', 'RSA': 'Africa', 'RUS': 'Europe',
    'SCG': 'Europe', 'SCO': 'Europe', 'SEN': 'Africa', 'SLV': 'North America', 'SRB': 'Europe',
    'SUI': 'Europe', 'SVK': 'Europe', 'SVN': 'Europe', 'SWE': 'Europe', 'TCH': 'Europe',
    'TOG': 'Africa', 'TRI': 'North America', 'TUN': 'Africa', 'TUR': 'Europe', 'UAE': 'Asia',
    'UKR': 'Europe', 'URS': 'Europe', 'URU': 'South America', 'USA': 'North America', 'WAL': 'Europe',
    'YUG': 'Europe', 'ZAI': 'Africa'
}
data['region'] = data['team_initials'].map(region_mapping).fillna('Other')
le_region = LabelEncoder()
data['region_encoded'] = le_region.fit_transform(data['region'])

# Match intensity
data['match_intensity'] = (data['num_yellow_cards'] + 2 * data['num_red_cards'] + data['num_substitutions']) / data['num_matches']

# Team-level features
data['win_rate'] = data['wins'] / data['matches_played']
data['goal_difference'] = data['avg_goals_scored'] - data['avg_goals_conceded']

# Player-focused features
data['goal_scorers_ratio'] = data['num_players_scored'] / data['num_matches']
data['disciplinary_score'] = (data['num_yellow_cards'] + 2 * data['num_red_cards']) / data['num_matches']

# Step 6: Create target variable
data['high_goals'] = (data['total_goals'] > 7).astype(int)
y = data['high_goals']
print("Target distribution:")
print(data['high_goals'].value_counts())

# Step 7: Prepare features
features = ['num_matches', 'num_players_scored', 'avg_starters_per_match', 'num_substitutions', 
            'num_yellow_cards', 'num_red_cards', 'num_unique_captains', 'num_unique_goalkeepers', 
            'region_encoded', 'match_intensity', 'win_rate', 'goal_difference', 
            'goal_scorers_ratio', 'disciplinary_score']
X = data[features]

# Step 8: Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
valid_features = [f for f, valid in zip(features, imputer.statistics_) if not np.isnan(valid)]
valid_indices = [i for i, f in enumerate(features) if f in valid_features]
X_imputed = X_imputed[:, valid_indices]

# Step 9: Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42, stratify=y)
except ValueError as e:
    print(f"Stratified split failed: {e}. Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Step 10: Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Step 11: Scale features
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Step 12: Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'criterion': ['gini', 'entropy']
}
dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_smote_scaled, y_train_smote)
print(f"\nBest Parameters: {grid_search.best_params_}")
best_dt_clf = grid_search.best_estimator_

# Step 13: Cross-validation
cv_scores = cross_val_score(best_dt_clf, X_train_smote_scaled, y_train_smote, cv=5, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Average CV F1 Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Step 14: Evaluate on test set
y_pred = best_dt_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 15: Visualize Decision Tree
dot_data = export_graphviz(best_dt_clf, out_file=None, 
                           feature_names=valid_features, 
                           class_names=['Low Goals', 'High Goals'], 
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('/home/seed/PROJECT/output/player_decision_tree', format='pdf', cleanup=True)
print("Decision tree saved as /home/seed/PROJECT/output/player_decision_tree.pdf")

# Step 16: Plot feature importance
plt.figure(figsize=(10, 5))
plt.bar(valid_features, best_dt_clf.feature_importances_)
plt.xticks(rotation=45)
plt.title('Feature Importance (Decision Tree)')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('/home/seed/PROJECT/output/player_feature_importance_dt.png')
plt.close()
print("Feature importance plot saved as /home/seed/PROJECT/output/player_feature_importance_dt.png")

# Step 17: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Goals', 'High Goals'], 
            yticklabels=['Low Goals', 'High Goals'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/home/seed/PROJECT/output/player_confusion_matrix_dt.png')
plt.close()
print("Confusion matrix plot saved as /home/seed/PROJECT/output/player_confusion_matrix_dt.png")

# Step 18: Plot ROC Curve
y_pred_prob = best_dt_clf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.grid(True)
plt.savefig('/home/seed/PROJECT/output/player_roc_curve_dt.png')
plt.close()
print(f"ROC Curve saved as /home/seed/PROJECT/output/player_roc_curve_dt.png")
print(f"AUC Score: {auc_score:.2f}")

# Step 19: Save predictions for MapReduce
predictions = pd.DataFrame({
    'team': data['team'],
    'team_initials': data['team_initials'],
    'actual_high_goals': y,
    'predicted_high_goals': best_dt_clf.predict(X_imputed)
})
predictions.to_csv('/home/seed/PROJECT/output/player_predictions.csv', index=False)
print("Predictions saved as /home/seed/PROJECT/output/player_predictions.csv")