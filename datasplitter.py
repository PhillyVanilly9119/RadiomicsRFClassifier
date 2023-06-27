import random
import pandas as pd

class DataSplitter():
    def __init__(self, frame) -> None:
        self.in_frame = frame
        
    def return_percentage_split(self):
        pass
    
    
    def split_data_train_test(self, classification_key: str, percentage: int):
        """This function takes the entire "df_input" frame and 
        1. sorts them after either of the
        classification keys 'Diagnosis', 'TissueType', 'IDHStatus'. 
        2. counts the # of different cases in the "classification_key" and loops through them,
        3. sorting out a percentage, accorting to "percentage" of unique Patient's samples (always 5)
        4. the split data frames are returend as a training and a test set.

        Args:
            df_input (pd.DataFrame): data frame containing all information of the study cohort 
            classification_key (str): either of the differtation metrics: 'Diagnosis', 'TissueType', 'IDHStatus'
            percentage (int, optional): Amount of data of every category that gets excluded for testing.. Defaults to 20.

        Raises:
            IndexError: if (at least) one doubled entry with index sorting occured (goal is to create mutually exlcusive index lists)
            Exception: if (at least) one doubled entry in the returned pd.DataFrame() occured

        Returns:
            pd.DataFrame(): test "df_test_data", and training "df_training_data" pd.DataFrames()
        """
        # pre-alloc of DataFrame()s that are returned
        df_test_data = pd.DataFrame()
        df_training_data = pd.DataFrame()
        all_test_idxs = []
        
        # loop through diagnostic category - max. of 5
        for _, val in enumerate(self.in_frame[classification_key].unique()): # 2. loop through the different categories (diagnosis) in "classification_key"
            self.in_frame
            assert self.in_frame[classification_key].value_counts()[val] % 5 == 0 # sanity check
            num_unique_patIDs = self.in_frame[classification_key].value_counts()[val] // 5 # find all unique PatIDs incurrent val
            num_excluded_patIDs_testing = int(num_unique_patIDs * (percentage / 100)) # calculate how many patIDs should be excluded for testing
            idxs_all_unique_patIDs = [val for _, val in enumerate(self.in_frame[self.in_frame[classification_key] == val].index) if val % 5 == 0]
            # pick random indices of patient patIDs accoring to the percentage of those that are being dropped               
            if percentage == 0: # if 0 is input, 2 random samples are excluded for testing
                idxs_random_patIDs = random.sample(idxs_all_unique_patIDs, 1)
            else:
                idxs_random_patIDs = random.sample(idxs_all_unique_patIDs, num_excluded_patIDs_testing)
            # find indices of entires that should be excluded from training for testing
            test_idxs = [] # indices of the rows containing all five samples of the patIDs that should be excluded
            
            for elem in idxs_random_patIDs:
                for add_one in range(5):
                        test_idxs.append(elem + add_one) # cover all 5 samples of the randomly selected patients that should be excluded
                        all_test_idxs.append(elem + add_one)
                            
            print(f"[INFO:] According to percentate ({percentage}%), {len(idxs_random_patIDs)} patient(s) (5 samples each) are excluded from the total of {num_unique_patIDs} samples for testing")
            print(f"in for the classification task: \"{classification_key}\" and diagnosis value (of the classificaiton task): \"{val}\".")
            
        training_idxs = [i for i in range(self.in_frame.shape[0]) if i not in all_test_idxs]
        if any(i in test_idxs for i in training_idxs):
            raise IndexError
            
        # append test data to frame
        df_test_data = df_test_data.append(self.in_frame.iloc[all_test_idxs])
        # append training data to frame
        df_training_data = df_training_data.append(self.in_frame.iloc[training_idxs])
        
        if any(i in list(test_idxs) for i in list(training_idxs)):
            raise IndexError
        
        return df_test_data, df_training_data
    
    
    def return_fixed_num_per_category(self):
        pass
    
    def return_random_mixed_split(self):
        pass