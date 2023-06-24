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
