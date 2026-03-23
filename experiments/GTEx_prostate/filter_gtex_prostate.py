import pandas as pd

# Load your GTEx_Portal_Prostate.csv
df = pd.read_csv('/local/data/magicscan/HnE/GTEx_prostate/GTEx_Portal_Prostate.csv')
df['Pathology Notes'] = df['Pathology Notes'].fillna('')

def classify(notes):
    notes = str(notes).lower()
    if any(x in notes for x in ['autolyzed', 'autolzyed', 'autolysis', 'sloughed', 'sloughing', 'no prostate', 'pin']):
        return 'Discard', -1
    if any(x in notes for x in ['adenocarcinoma', 'carcinoma', 'gleason']):
        return 'Cancer', 1
    return 'Benign', 0

df['class_name'], df['label'] = zip(*df['Pathology Notes'].apply(classify))
df[['Tissue Sample ID', 'class_name', 'label', 'Pathology Notes']].to_csv('/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv', index=False)