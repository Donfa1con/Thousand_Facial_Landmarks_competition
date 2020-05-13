import pandas as pd

if __name__ == "__main__":
    subm1 = pd.read_csv('subs/subs_1.csv', index_col='file_name')
    subm2 = pd.read_csv('subs/subs_2.csv', index_col='file_name')
    res_subm = (subm1 + subm2) / 2 + 0.5
    res_subm[res_subm.columns].astype(int).reset_index().to_csv('subs/mean_sub.csv', index=False)
