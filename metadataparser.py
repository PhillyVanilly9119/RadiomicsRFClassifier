import os, sys

class MetaDataParser():
    
    def __init__(self, working_dir):
        working_dir = self.working_dir
        
    
    def _get_data_file_list(self):
        """
        Returns:
            list: of *.CSV files containing the TF-data
        """
        return ["1_Tumor_diagnosis_data.txt",
                "2_Tissue_type_diagnosis_data.txt",
                "3_Glioma_diagnosis_tissue_type_data.txt",
                "4_Glioma_IDH_data.txt"] 

    def _get_key_type(self):
        """
        Returns:
            list: list of prediction tasks
        """
        return ['Diagnosis',
                'TissueType',
                'IDHStatus']  

    def _get_feature_group_list(self):
        """
        Returns:
            list: of files containing the features
        """
        return ["feature_group_1.json",
                "feature_group_2.json",
                "feature_group_3.json",
                "feature_group_4.json"] 
 
    def _get_experiment_meta_data(self):
        """List of lists containing all combinations for all 13 experiments -> see "howto.txt"
        _trainings[n][0] -> path to data file (CSV)
        _trainings[n][1] -> classification key
        _trainings[n][2] -> path to feature list file (JSON)
        _trainings[n][3] -> file name to save model, trained on all number of TF-values
        _trainings[n][4] -> file name to save model, trained on essential number of TF-values

        Returns:
            list: _trainings[iteration][detail]
        """
        _key_list = self._get_key_type()
        _data_file_list = self._get_data_file_list()
        _feature_group_list = self._get_feature_group_list()
        
        return [
        # "Tumor Diagnosis 1" 
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[0]), 
        _key_list[0],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[2]),
        "RFC_Model_allFeatures_TumorDiagnosis_PyRad_HRALA.joblib",
        "RFC_Model_relevantFeatures_TumorDiagnosis_PyRad_HRALA.joblib"],
        # "Tissue Type 1"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[1]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[0]),
        "RFC_Model_allFeatures_TissueType_PyRad.joblib",
        "RFC_Model_relevantFeatures_TissueType_PyRad.joblib"],
        # "Tissue Type 2"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[1]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[1]),
        "RFC_Model_allFeatures_TissueTypePyRad_IntraOPALA.joblib",
        "RFC_Model_relevantFeatures_TissueTypePyRad_IntraOPALA.joblib"],
        # "Tissue Type 3"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[1]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[2]),
        "RFC_Model_allFeatures_TissueType_PyRad_HRALA.joblib",
        "RFC_Model_relevantFeatures_TissueType_PyRad_HRALA.joblib"],
        # "Glioma 1"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[0]),
        "RFC_Model_allFeatures_Glioma_TissueType_PyRad.joblib",
        "RFC_Model_relevantFeatures_Glioma_TissueType_PyRad.joblib"],
        # "Glioma 2"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[3]),
        "RFC_Model_allFeatures_Glioma_TissueType_HRALA.joblib",
        "RFC_Model_relevantFeatures_Glioma_TissueType_HRALA.joblib"],
        # "Glioma 3"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[1],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[2]),
        "RFC_Model_allFeatures_Glioma_TissueType_PyRAD_HRALA.joblib",
        "RFC_Model_relevantFeatures_Glioma_TissueType_PyRAD_HRALA.joblib"],
        # "Glioma 4"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[0],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[0]),
        "RFC_Model_allFeatures_Diagnosis_PyRad.joblib",
        "RFC_Model_relevantFeatures_Diagnosis_PyRad.joblib"],
        # "Glioma 5"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[0],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[3]),
        "RFC_Model_allFeatures_Diagnosis_HRALA5.joblib",
        "RFC_Model_relevantFeatures_Diagnosis_HRALA.joblib"],
        # "Glioma 6"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[2]), 
        _key_list[0],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[2]),
        "RFC_Model_allFeatures_Diagnosis_PyRad_HRALA.joblib",
        "RFC_Model_relevantFeatures_Diagnosis_PyRad_HRALA.joblib"],
        # "IDH Status Prediciton 1"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[3]), 
        _key_list[2],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[0]),
        "RFC_Model_allFeatures_IDHStatus_PyRad.joblib",
        "RFC_Model_relevantFeatures_IDHStatus_PyRad.joblib"],
        # "IDH Status Prediciton 2"
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[3]), 
        _key_list[2],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[3]),
        "RFC_Model_allFeatures_IDHStatus_HRALA.joblib",
        "RFC_Model_relevantFeatures_IDHStatus_HRALA.joblib"],
        # "IDH Status Prediciton 3": 
        [os.path.join(self.working_dir, "CSV_Files", _data_file_list[3]), 
        _key_list[2],
        os.path.join(self.working_dir, "Feature_Names", _feature_group_list[2]),
        "RFC_Model_allFeatures_IDHStatus_PyRad_HRALA.joblib",
        "RFC_Model_relevantFeatures_IDHStatus_PyRad_HRALA.joblib"]
        ]    