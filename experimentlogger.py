import os
from datetime import datetime

class ExperimentLogger():
    
    def __init__(self, exp_path) -> None:
        self.create_folder_for_curr_exp()
        self.main_path_exp = exp_path
    
    def create_folder_for_curr_exp(self) -> None:
        curr_datetime = datetime.now()
        return
    
    def create_log_file_for_curr_exp(self) -> None:
        pass
    
    def save_message_to_current_exp_file(self, message) -> None:
        pass




    def log_grid_search_predictions(grid, file_path):
        grid_predictions = grid.predict(X_test)
        with open(file_path, 'a') as f:
            f.write(f"\n[RESULT:] model reached accuracies of (after grid search):\n{classification_report(y_test, grid_predictions)}\n")
            f.write(f"\n[RESULT:] confusion matrix of model:\n{confusion_matrix(y_test, grid_predictions)}")
            f.write(f"\n[RESULT:] weighted F1-score of Decision Tree classifier on train dataset: {grid.score(X_train, y_train):.2f}\n")
            f.write(f"\n[RESULT:] weighted F1-score of Decision Tree classifier on testset: {grid.score(X_test, y_test):.3f}\n")
            f.write(f"[INFO:] Best set of params for current experiment were: {grid.best_params_}\n")