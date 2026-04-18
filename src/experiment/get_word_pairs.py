import pandas as pd


def get_stratified_simlex_pairs(filepath, n, word_to_centroid_indices, seed=42):
    df = pd.read_csv(filepath, sep="\t")

    vocab = set(word_to_centroid_indices.keys())

    # Keep only rows where both words exist in the dictionary
    df = df[df["word1"].isin(vocab) & df["word2"].isin(vocab)]

    low = df[df["SimLex999"] < 3.33]
    mid = df[(df["SimLex999"] >= 3.33) & (df["SimLex999"] < 6.66)]
    high = df[df["SimLex999"] >= 6.66]

    # Sample equally from each
    s1 = low.sample(n // 3, random_state=seed)
    s2 = mid.sample(n // 3, random_state=seed)
    s3 = high.sample(n // 3, random_state=seed)

    sampled_df = pd.concat([s1, s2, s3])
    return list(zip(sampled_df["word1"], sampled_df["word2"], sampled_df["SimLex999"]))
