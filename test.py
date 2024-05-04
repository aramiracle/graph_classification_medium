import pandas as pd

directory = "models/PROTEINS/train_log.csv"

df = pd.read_csv(directory)

print(df.sort_values(by=['Test Accuracy', 'Average Loss']))