def num_cat_split(df, ratio_threshold=0.05):
    num_cols = []
    cat_cols = []
    for col in df.columns:
        n_unique = df[col].nunique()
        ratio = n_unique / len(df)

        # Determine column type
        if ratio <= ratio_threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols

def print_col_percentage(df, col):
  print(df[col].value_counts() / len(df))