
# import pandas as pd

# # Load the file
# participants = pd.read_csv("D:\\CompBioFInal\\participants.tsv", sep="\t")

# # Show all subjects and their groups
# print(participants[["participant_id", "Group", "Age", "MMSE"]])

# # Count each group
# print("\nGroup counts:")
# print(participants["Group"].value_counts())

# # Separate into three groups
# ad_subjects  = participants[participants["Group"] == "A"]
# cn_subjects  = participants[participants["Group"] == "C"]
# ftd_subjects = participants[participants["Group"] == "F"]

# print(f"\nAlzheimer's patients: {len(ad_subjects)}")
# print(f"Healthy controls:     {len(cn_subjects)}")
# print(f"FTD patients:         {len(ftd_subjects)}")

# # Show AD subjects specifically
# print("\nAD subjects:")
# print(ad_subjects[["participant_id", "Age", "MMSE"]])

import os

DATASET_PATH = "D:\\CompBioFInal"  # change this to wherever your folder is

print("=== TOP LEVEL ===")
print(os.listdir(DATASET_PATH))

print("\n=== FIRST SUBJECT FOLDER ===")
sub_path = os.path.join(DATASET_PATH, "sub-001")
if os.path.exists(sub_path):
    print(os.listdir(sub_path))
    eeg_path = os.path.join(sub_path, "eeg")
    if os.path.exists(eeg_path):
        print("\n=== EEG FOLDER ===")
        print(os.listdir(eeg_path))
else:
    print("sub-001 folder not found")
    print("Folders found:", os.listdir(DATASET_PATH))