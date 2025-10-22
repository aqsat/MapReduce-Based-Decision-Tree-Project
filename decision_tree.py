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
team_matches = pd.read_csv('/home/seed/PROJECT/output/team_matches.txt', sep='\t', names=['team', 'matches_played'])
team_outcomes = pd.read_csv('/home/seed/PROJECT/output/team_outcomes.txt', sep='\t', 
                            names=['team', 'wins', 'draws', 'losses'])
team_avg_goals = pd.read_csv('/home/seed/PROJECT/output/team_avg_goals.txt', sep='\t', 
                             names=['team', 'avg_goals_scored', 'avg_goals_conceded', 'matches_played'])

# Step 2: Merge datasets
data = pd.merge(team_matches, team_outcomes, on='team', how='left')
data = pd.merge(data, team_avg_goals[['team', 'avg_goals_scored', 'avg_goals_conceded']], on='team', how='left')

# Step 3: Debug merges
print("Missing values after merges:")
print(data[['wins', 'draws', 'losses', 'avg_goals_scored', 'avg_goals_conceded']].isna().sum())

# Step 4: Feature Engineering
# Team-level features
data['win_rate'] = data['wins'] / data['matches_played']
data['loss_rate'] = data['losses'] / data['matches_played']
data['draw_rate'] = data['draws'] / data['matches_played']
data['goal_difference'] = data['avg_goals_scored'] - data['avg_goals_conceded']
data['total_goals'] = data['avg_goals_scored'] * data['matches_played']

# Region mapping
region_mapping = {
    'algeria': 'Africa', 'angola': 'Africa', 'argentina': 'South America', 'australia': 'Asia', 
    'austria': 'Europe', 'belgium': 'Europe', 'bosnia and herzegovina': 'Europe', 'bolivia': 'South America',
    'brazil': 'South America', 'bulgaria': 'Europe', 'cameroon': 'Africa', 'canada': 'North America',
    'chile': 'South America', 'china pr': 'Asia', 'côte d\'ivoire': 'Africa', 'colombia': 'South America',
    'costa rica': 'North America', 'croatia': 'Europe', 'cuba': 'North America', 'czech republic': 'Europe',
    'denmark': 'Europe', 'ecuador': 'South America', 'egypt': 'Africa', 'england': 'Europe', 'spain': 'Europe',
    'france': 'Europe', 'germany': 'Europe', 'ghana': 'Africa', 'greece': 'Europe', 'haiti': 'North America',
    'honduras': 'North America', 'hungary': 'Europe', 'indonesia': 'Asia', 'republic of ireland': 'Europe',
    'iran': 'Asia', 'iraq': 'Asia', 'israel': 'Asia', 'italy': 'Europe', 'jamaica': 'North America',
    'japan': 'Asia', 'korea republic': 'Asia', 'saudi arabia': 'Asia', 'kuwait': 'Asia', 'morocco': 'Africa',
    'mexico': 'North America', 'netherlands': 'Europe', 'nigeria': 'Africa', 'northern ireland': 'Europe',
    'norway': 'Europe', 'new zealand': 'Oceania', 'paraguay': 'South America', 'peru': 'South America',
    'poland': 'Europe', 'portugal': 'Europe', 'korea dpr': 'Asia', 'romania': 'Europe', 'south africa': 'Africa',
    'russia': 'Europe', 'serbia': 'Europe', 'scotland': 'Europe', 'senegal': 'Africa', 'el salvador': 'North America',
    'switzerland': 'Europe', 'slovakia': 'Europe', 'slovenia': 'Europe', 'sweden': 'Europe', 'czechoslovakia': 'Europe',
    'togo': 'Africa', 'trinidad and tobago': 'North America', 'tunisia': 'Africa', 'turkey': 'Europe',
    'united arab emirates': 'Asia', 'ukraine': 'Europe', 'soviet union': 'Europe', 'uruguay': 'South America',
    'united states': 'North America', 'wales': 'Europe', 'yugoslavia': 'Europe', 'zaire': 'Africa'
}
data['region'] = data['team'].map(region_mapping).fillna('Other')
le_region = LabelEncoder()
data['region_encoded'] = le_region.fit_transform(data['region'])

# Step 5: Create target variable
data['high_performance'] = ((data['win_rate'] > 0.5) | (data['goal_difference'] > 0.5)).astype(int)
y = data['high_performance']  # Assign y
print("Target distribution:")
print(data['high_performance'].value_counts())

# Step 6: Handle missing values and feature selection
features = ['matches_played', 'wins', 'draws', 'losses', 'avg_goals_scored', 'avg_goals_conceded', 
            'loss_rate', 'draw_rate', 'total_goals', 'region_encoded']
X = data[features]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Identify valid features
valid_features = [f for f, valid in zip(features, imputer.statistics_) if not np.isnan(valid)]
valid_indices = [i for i, f in enumerate(features) if f in valid_features]
X_imputed = X_imputed[:, valid_indices]

# Step 7: Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42, stratify=y)
except ValueError as e:
    print(f"Stratified split failed: {e}. Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Step 8: Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Step 9: Scale features
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Step 10: Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 6],
    'criterion': ['gini', 'entropy']
}
dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_smote_scaled, y_train_smote)
print(f"\nBest Parameters: {grid_search.best_params_}")
best_dt_clf = grid_search.best_estimator_

# Step 11: Cross-validation
cv_scores = cross_val_score(best_dt_clf, X_train_smote_scaled, y_train_smote, cv=5, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Average CV F1 Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Step 12: Evaluate on test set
y_pred = best_dt_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Visualize Decision Tree
dot_data = export_graphviz(best_dt_clf, out_file=None, 
                           feature_names=valid_features, 
                           class_names=['Low Performance', 'High Performance'], 
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('/home/seed/PROJECT/output/team_decision_tree', format='pdf', cleanup=True)
print("Decision tree saved as /home/seed/PROJECT/output/team_decision_tree.pdf")

# Step 14: Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=best_dt_clf.feature_importances_, y=valid_features)
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('/home/seed/PROJECT/output/team_feature_importance_dt.png')
plt.close()
print("Feature importance plot saved as /home/seed/PROJECT/output/team_feature_importance_dt.png")

# Step 15: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Performance', 'High Performance'], 
            yticklabels=['Low Performance', 'High Performance'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/home/seed/PROJECT/output/team_confusion_matrix_dt.png')
plt.close()
print("Confusion matrix plot saved as /home/seed/PROJECT/output/team_confusion_matrix_dt.png")

# Step 16: Plot ROC Curve
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
plt.savefig('/home/seed/PROJECT/output/team_roc_curve_dt.png')
plt.close()
print(f"ROC Curve saved as /home/seed/PROJECT/output/team_roc_curve_dt.png")
print(f"AUC Score: {auc_score:.2f}")

# Step 17: Save predictions for MapReduce
predictions = pd.DataFrame({
    'team': data['team'],
    'actual_high_performance': y,
    'predicted_high_performance': best_dt_clf.predict(X_imputed)
})
predictions.to_csv('/home/seed/PROJECT/output/team_predictions.csv', index=False)
print("Predictions saved as /home/seed/PROJECT/output/team_predictions.csv")