import joblib
import os, sys
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split

### load model
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

if not file_path.endswith('.joblib'):
    print("You have seem to selected a file which does not contain a trained network")
rfc_model = joblib.load(file_path)
