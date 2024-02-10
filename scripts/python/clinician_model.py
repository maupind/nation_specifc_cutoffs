import pyarrow.feather as feather
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import xgboost
import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.monte_carlo import qExpectedImprovement
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import shap
import joblib

# Replace with path to clinician data
clin = feather.read_feather('/path/to/your/ind_clin.feather')

## Choose rater ID
#Rater 1 - 7049
#Rater 2 - 7021
#Rater 3 - 7038
#Rater 4 - 7033
#Rater 5 - 7043
#Rater 6 - 7020
#Rater 7 - 7046
#Rater 8 - 7042
#Rater 9 - 7023
clin0 = clin[clin['r1raterid'] == ]


# Setting up Predictors and Result
X = clin0.drop('cdr', axis = 1).copy()
#X = X.drop('r1raterid', axis = 1).copy()
X = X.drop('prim_key', axis = 1).copy()
X = X.drop('hh1rural', axis = 1).copy()
X = X.drop('r1illiterate', axis = 1).copy()
X = X.drop('r1cdr_incon', axis = 1).copy()
X = X.drop('r1raterid', axis = 1).copy()
X.head()

y = clin0['cdr'].copy()
y.head()

# Changing Data types
le = LabelEncoder()
y = le.fit_transform(y)

columns_to_transform = [
    'ragender.x',
    #'r1lang_d',
    'raedyrs.x',
    'r1mstat',
    #'r1lbrf_l',
    'r1alzdeme',
    'r1psyche',
    'r1stroke',
    'r1urinae',
    'r1hibpe',
    'r1hearte',
    'r1diabe',
    #'r1mobilc',
    #'r1mobilca',
    #'r1mobilsev_l',
    #'r1mobilseva_l'
]

X_transform = X.copy()

for column in columns_to_transform:
    le = LabelEncoder()
    X_transform[column] = le.fit_transform(X[column])

for column in columns_to_transform:
    X_transform[column] = X_transform[column].where(~X[column].isna(), X[column])
    X_transform[column] = X_transform[column].fillna(-100)

# Lists to collect SHAP values and AUC scores for each fold
all_shap_values = []
all_val_data = []
all_thresholds = []
best_model = None

X_transform = X_transform.rename(
    columns = {
        'r1agey.x' : 'Age',
        'ragender.x' : 'Gender',
        'r1mstat' : 'Marital Status',
        'r1jcocc_l' : 'Occupation',
        'r1alzdeme' : 'Dementia and Alzheimers History',
        'r1psyche' : 'Depression History',
        'r1stroke' : 'Stroke History',
        'r1urinae' : 'Urinary Incontinence History',
        'r1hearte' : 'Heart Disease History',
        'r1diabe' : 'Diabetes History',
        'r1hibpe' : 'High Blood Pressure History',
        'r1systo.x' : 'Mean Systolic Blood Pressure',
        'r1diasto.x' : 'Mean Diastolic Blood Pressure',
        'r1i_memory' : 'Self-Rated Memory',
        'r1i_compmem' : 'Compared Memory Status (2 years ago)',
        'r1i_hear' : 'Sensory Impairment',
        'r1hmse_score' : 'Hindi Mental State Exam',
        'r1hmse_scorz' : 'Stdized Hindi Mental State Exam',
        'r1cesd10.x' : 'Center for Epidemiological Studies - Depression',
        'r1mindtsl' : 'Trouble Concentrating',
        'r1depresl' : 'Felt Depressed',
        'r1effortl' : 'Everything an Effort',
        'r1ftiredl' : 'Felt Tired or Low Energy',
        'r1whappyl' : 'Was Happy',
        'r1flonel' : 'Felt Alone',
        'r1fsatisl' : 'Felt Satisifed Overall',
        'r1fearfll' : 'Felt Afraid of Something',
        'r1fhopel' : 'Felt Hopeful',
        'r1botherl' : 'Bothered by Little Things',
        'r1anx5' : 'Anxiety Inventory - Overall',
        'r1worst' : 'Anxiety Inventory - Worst Happening',
        'r1nerv' : 'Anxiety Inventory - Nervous',
        'r1tremb' : 'Anxiety Inventory - Hands Trembling',
        'r1fdying' : 'Anxiety Inventory - Fear of Dying',
        'r1faint' : 'Anxiety Inventory - Felt Faint',
        'r1inf_rel' : 'Informant Relationship',
        'rinf_care' : 'Whether Informant Provides Care',
        'r1inf_yrs' : 'Years Informant Knows Respondent',
        'r1inf_freq' : 'Frequency Informant Visit',
        'r1bl1score' : 'Blessed Test Part 1',
        'r1bl2score' : 'Blessed Test Part 2',
        'r1iqscore1' : 'IQCODE - Family/Friend Details',
        'r1iqscore2' : 'IQCODE - Recent Events',
        'r1iqscore3' : 'IQCODE - Recent Conversations',
        'r1iqscore4' : 'IQCODE - Address/Telephone #',
        'r1iqscore5' : 'IQCODE - Day and Month',
        'r1iqscore6' : 'IQCODE - Where Things Are Kept',
        'r1iqscore7' : 'IQCODE - Where to Find Things',
        'r1iqscore8' : 'IQCODE - Work Familiar Machines',
        'r1iqscore9' : 'IQCODE - New Gadget/Machine',
        'r1iqscore10' : 'IQCODE - New Things in General',
        'r1iqscore11' : 'IQCODE - Story in Book/TV',
        'r1iqscore12' : 'IQCODE - Making Decisions',
        'r1iqscore13' : 'IQCODE - Handling Money',
        'r1iqscore14' : 'IQCODE - Handling Financial Matters',
        'r1iqscore15' : 'IQCODE - Handling Arithmetic',
        'r1iqscore16' : 'IQCODE - Reason Things Through',
        'r1jormscore' : 'IQCODE - Average',
        'r1scis' : 'Cognition - Scissors',
        'r1coconut' : 'Cognition - Coconut',
        'r1prime' : 'Cognition - Prime Minister',
        'r1tics_score' : 'Telephone Interview Cognitive Status',
        'r1ef_palm' : 'Repeat Palm-Up, Palm-Down',
        'r1ef_clench' : 'Clenched Extended Hand Movement',
        'r1ef_fist' : 'Fist-Side-Palm Test',
        'r1ef_score' : 'Hand Sequencing Score',
        'r1tt_crcl' : 'Identify Circle',
        'r1tt_sq' : 'Identify Square',
        'r1tt_dmnd' : 'Identify Diamond',
        'r1tt_blckcrcl' : 'Identify Black Circle/Diamond',
        'r1ttblsqr' : 'Identify Blue/Yellow Square',
        'r1tt_yldmnd' : 'Identify Yellow Diamond, Blue Circle',
        'r1tt_ylsqr' : 'Identify Yellow Square, Black Circle',
        'r1tt_score' : 'Token Test Score',
        'r1jp_animl' : 'Animal Similarites',
        'r1jp_flwr' : 'Flower Similarities',
        'r1jplie' : 'Lie/Mistake Differences',
        'r1jp_river' : 'River/Pond Differences',
        'r1jp_rupee1' : 'Coins for One Rupee',
        'r1jp_rupee2' : 'Coins for 6.5 Rupees',
        'r1jp_fndkid' : 'Find Lost Child',
        'r1sim_score' : 'Similarity and Differences Score',
        'r1pro_score' : 'Problem Solving Score',
        'r1ds_back' : 'Digit Span Backward',
        'r1ds_for' : 'Digit Span Forward',
        'r1csi1' : 'General Decline in Mental Function',
        'r1csi2' : 'Remebering Things is a Problem',
        'r1csi3' : 'Forget Where Put Things',
        'r1csi4' : 'Forget Where Things Are Kept',
        'r1csi5' : 'Forget Names of Friends',
        'r1csi6' :  'Forget Names of Family',
        'r1csi7' : 'Forget Statement while Talking',
        'r1csi8' : 'Difficulty Finding Right Words',
        'r1csi9' : 'Using Wrong Words',
        'r1csi10' : 'Tend to Talk About What Happened Long Ago',
        'r1csi11' : 'Forget When Last Saw Informant',
        'r1csi12' : 'Forget What Happened the Day Before',
        'r1csi13' : 'Forget Where They Are',
        'r1csi14' : 'Get Lost in the Community',
        'r1csi15' : 'Get Lost in Home',
        'r1elbow' : 'Cognition - Elbow',
        'r1hammer' : 'Cognition - Hammer',
        'r1store' : 'Cognition - Store',
        'r1point' : 'Cognition - Point',
        'r1word1' : 'Word List Trial 1',
        'r1word2' : 'Word List Trial 2',
        'r1word3' : 'Word List Trial 3',
        'r1word_total' : 'Word List Total',
        'r1wre_org' : 'Word List Recognition Original',
        'r1wre_foil' : 'Word List Recognition New',
        'r1wre_score' : 'Word List Recognition Score',
        'r1csid_score' : 'Coomunity Screeing Interview of Dementia',
        'r1rv_score' : 'Ravens Test',
        'r1ser7.x' : 'Serial 7s',
        'r1go_score1' : 'Go-no-go Trial 1',
        'r1go_score2' : 'Go-no-go Trial 2',
        'r1go_score' : 'Go-no-go Total',
        'r1sc_anw' : 'Symbol Cancellation',
        'r1verbal' : 'Verbal Fluency Animal Naming - Correct',
        'r1verbal_inc' : 'Verbal Fluency Animal Naming - Incorrect',
        'r1verbal_prb' : 'Verbal Fluency Animal Naming - Problem',
        'r1adla_d' : 'Total Activities of Daily Living',
        'r1walkra.x' : 'Difficulty Walking',
        'r1batha.x' : 'Difficulty Bathing',
        'r1dressa.x' : 'Difficulty Dressing',
        'r1eata.x' : 'Difficulty Eating',
        'r1beda.x' : 'Difficulty Getting In/Out Bed',
        'r1toilta.x' : 'Difficulty Using Toilet',
        'r1iadltot1_d' : 'Total Instrumental Activities of Daily Living',
        'r1phonea.x' : 'Difficulty Making Phone Calls',
        'r1moneya.x' : 'Difficulty Handling Money',
        'r1medsa.x' : 'Difficulty Taking Medications',
        'r1shopa.x' : 'Difficulty Shopping',
        'r1mealsa.x' : 'Difficulty Preparing Meals',
        'r1housewka.x' : 'Difficulty Doing Housework',
        'r1geta.x' : 'Difficulty Getting Around',
        'r1walk100a' : 'Difficulty Walking 100 yards',
        'r1sita' : 'Difficulty Sitting 2 Hours',
        'r1chaira' : 'Difficulty Getting Up From Chair',
        'r1stoopa' : 'Difficulty Stooping/Kneeling',
        'r1armsa' : 'Difficulty Extending Arms',
        'r1pusha' : 'Difficulty Pushing Large Objects',
        'r1lifta' : 'Difficulty Lifting Large Objects',
        'r1dimea' : 'Difficulty Picking Up a Coin',
        'r1clim1a' : 'Difficulty Climbing 1 Flight of Stairs',
        'r1act_tv' : 'Time Spent Watching TV',
        'r1act_read' : 'Time Spent Reading',
        'r1act_chor' : 'Time Spent Doing Chores',
        'r1act_comp' : 'Time Spent Using Computer',
        'r1act_nap' : 'Time Spent Napping',
        'r1act_meal' : 'Whether Prepares Hot Meals',
        'r1act_trav' : 'Whether Can Travel Alone',
        'r1act_pubt' : 'Whenther Uses Public Transport',
        'r1act_work' : 'How Often Goes to Work/Volunteers',
        'r1act_stor' : 'How Often Goes to Store/Market',
        'r1act_walk' : 'How Often Goes for Walks',
        'r1act_spor' : 'How Often Does Yoga/Exercise',
        'r1ten1' : '10-66 House Hold Chores',
        'r1ten2' : '10-66 Special Skill or Hobby',
        'r1ten3' : '10-66 Handle Money',
        'r1ten4' : '10-66 Adjusting to Change',
        'r1ten5' : '10-66 Think and Reason',
        'mh046' : 'Computing Sale in Shop',
        'mh047' : 'Computing Lottery',
        'ht009' : 'History Neurological Problems',
        'r1bm_imm_d' : 'Brave Man Immediate 10 Point Score',
        'r1bm_immex' : 'Brave Man Immediate Summary Score Exact',
        'r1bm_imm' : 'Brave Man Immediate Summary Score',
        'r1bm_recl' : 'Brave Man Recall Summary Score',
        'r1bm_s4' : 'Brave Man Recall Story Point 4',
        'raedyrs.x' : 'Years of Education',
        'r1bm_s7' : 'Brave Man Recall Story Point 7',
        'r1cpr_circle' : 'Drawing Circle Recall',
        'r1cp_circle' : 'Drawing Circle Score',
        'r1cp_rectan' : 'Drawing Rectangle Score',
        'r1cpr_rectan' :'Drawing Rectangle Recall',
        'r1cp_cube' : 'Drawing Cube Score',
        'r1cpr_cube' : 'Drawing Cube Recall',
        'r1cp_diamon' : 'Drawing Diamond Score',
        'r1cpr_diamon' : 'Drawing Diamond Recall',
        'r1cp_score' : 'Constructional Praxis Score',
        'r1cpr_score' : 'Constructional Praxis Recall Score',
        'r1lmb_s16' : 'Robbery Story Point 16',
        'r1lmb_s8' : 'Robbery Story Point 8',
        'r1lmb_recl_d' : 'Robbery Story Delayed Recall',
        'r1lmb_imm' : 'Robbery Story Immediate Recall',
        'r1lmb_imm_d' : 'Robbery Story Immediate Recall with Approximation',
        'r1bm_recl_d' : 'Brave Man Story Immediate Recall with Approximation',
        'r1lmb_s22' : 'Robbery Storcy Point 22'
   }
)


# Cross-Validation - basically adjusted parameters by either keeping if on lower/middle or trying higher ones until settled on lower or middle value
search_spaces = {
    'learning_rate': (0.01, 0.3),  # These will be normalized in the optimization loop
   # 'max_depth': np.arange(3, 10, dtype=int),
    'gamma': (0, 1),
    'reg_lambda': (0.1, 10),
    'scale_pos_weight': (0.1, 10),
    'colsample_bytree': (0.5, 1),
    'subsample': (0.5, 1)
}

# Convert search_spaces dictionary into a list of tuples
bounds_list = [(v[0], v[1]) for k, v in search_spaces.items()]

# Convert bounds_list into a tensor
bounds_tensor = torch.tensor(bounds_list, dtype=torch.float64)


# Split the data into 80% training and 20% testing
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_transform, 
    y, 
    test_size=0.3, 
    random_state=33, 
    stratify=y
)

clf_xgb =  xgboost.XGBClassifier(objective = 'binary:logistic',
                                seed = 33)



# Initialize an empty training dataset
train_X = []
train_Y = []

def objective_function(params):
    # Set the XGBoost parameters
    params = {key: sublist[0].item() for key, sublist in zip(search_spaces.keys(), params)}
    clf_xgb.set_params(**params)
    # Train the XGBoost model on the training data for this fold
    clf_xgb.fit(X_train_fold, y_train_fold, 
                early_stopping_rounds=75,  # reduced for demonstration
                eval_metric='auc',
                eval_set=[(X_val_fold, y_val_fold)], 
                verbose=False)
    
    # Make predictions on the validation set and compute AUC
    predictions = clf_xgb.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, predictions)
    
    # Return the negative AUC since botorch aims to minimize
    return auc

#gp = SingleTaskGP(train_X=train_X, train_Y=train_Y)  # Initialize GP model
#ei = ExpectedImprovement(gp)  # Initialize Expected Improvement acquisition function


# StratifiedKFold ensures each fold is representative of class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)

# Bayesian optimization loop (replacing BayesSearchCV)
num_iterations = 50  # Set the number of iterations
best_auc = 0
best_params = None

# Running the Model

# Start 10-fold cross-validation
for train_idx, val_idx in skf.split(X_train_full, y_train_full):
    X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    train_X = []
    train_Y = []

    for i in range(num_iterations):
        # Draw a Sobol sample
        X = draw_sobol_samples(bounds=bounds_tensor, n=1, q=1).squeeze(0)

        # Unnormalize the parameters
        X_unnorm = unnormalize(X, bounds = bounds_tensor)

        # Compute objective
        objective_value = objective_function(X_unnorm)
        
        # Append to training data lists
        train_X.append(X)
        train_Y.append([objective_value])

    # Convert lists to tensors for GP model training
    train_X_tensor = torch.stack(train_X, dim=0)
    train_Y_tensor = torch.tensor(train_Y, dtype=torch.float64)
    train_Y_tensor = train_Y_tensor.unsqueeze(-1)

    # Compute the best AUC score from the optimization process
    fold_best_auc = max(train_Y)[0]  # Assuming train_Y contains the negative AUC values
    print(f"AUC for fold: {fold_best_auc}")
    
    gp = SingleTaskGP(train_X=train_X_tensor, train_Y=train_Y_tensor)
    ei = ExpectedImprovement(gp, best_f=fold_best_auc)

    if fold_best_auc > best_auc:
        best_auc = fold_best_auc
        best_params = {key: X_unnorm[0].tolist() for key in search_spaces.keys()}
        best_model = clf_xgb

    # Compute and store SHAP values for the validation fold
    explainer = shap.TreeExplainer(clf_xgb)
    shap_values = explainer.shap_values(X_val_fold)
    all_shap_values.append(shap_values)
        #Add
    all_val_data.append(X_val_fold)


# Stack all SHAP values and validation data
all_shap_values_stacked = np.vstack(all_shap_values)
all_val_data_stacked = pd.concat(all_val_data, axis=0)

# Display overall best AUC and corresponding parameters
print(f"Best AUC overall: {best_auc}")
print(f"Best parameters overall: {best_params}")
#AUC 0.92

# Retrain the model with the best parameters on the full 70% training data
best_model.fit(X_train_full, y_train_full)
 

y_test_pred = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_test_pred)
print(f"AUC on test set: {auc_score}")
#0.89

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold = 0.11
y_test_pred_binary = (y_test_pred > threshold).astype(int)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred_binary)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate sensitivity (recall)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

# Calculate specificity
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Display the results
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

#Confusion Matrix:
#[[1720  336]
# [  12  161]]
#Sensitivity (Recall): 0.9306
#Specificity: 0.8366

#Confusion Matrix:
#[[1837  219]
# [  12  161]]
#Sensitivity (Recall): 0.9306
#Specificity: 0.8935

# Assuming predicted_probabilities is the output from your trained XGBoost model
thresholds = np.linspace(0, 1, 1000)  # Adjust the number of thresholds as needed
best_thresholds = []

sensitivity = []
specificity = []

for threshold in thresholds:
    y_pred = (y_test_pred > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)  # You need to provide the true labels (y_true)
    sensitivity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    specificity.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))

youden_indices = np.array(sensitivity) + np.array(specificity) - 1
best_thresholds.append(thresholds[np.argmax(youden_indices)])

best_thresholds = np.array(best_thresholds)
print(best_thresholds)


# Save the best model for future use
#joblib.dump(best_model, "/home/danny/Documents/Projects/lasi/python/clinical_dx/values/ind_clin/best_xgb_clin12.pkl")

#joblib.dump(all_shap_values_stacked, '/home/danny/Documents/Projects/lasi/python/clinical_dx/values/ind_clin/clin12_shap.joblib')
#joblib.dump(all_val_data_stacked, '/home/danny/Documents/Projects/lasi/python/clinical_dx/values/ind_clin/clin12_values.joblib')
# Average SHAP values across all folds and plot summary
#avg_shap_values = np.mean(all_shap_values, axis=0) 

shap.initjs()
orig_map = plt.cm.get_cmap('copper')
reversed_map = orig_map.reversed()
shap.summary_plot(all_shap_values_stacked, all_val_data_stacked)  # Considering using larger datasets for a more global view
shap.summary_plot(all_shap_values_stacked, all_val_data_stacked, plot_type="bar")