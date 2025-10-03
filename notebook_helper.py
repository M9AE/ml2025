import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def split_url(url):
    '''transforme un url en splittant autour du second point.'''
    if not isinstance(url, str):
        return (np.nan, np.nan)

    url = url[7:-1]
    url_split = url.split(".")
    radical = url_split[0]
    extension = ".".join(url_split[1:])
    return (radical, extension)


def explicit_part_preparation(df):
    '''Prend en entrée un dataframe et prépare les N premières colonnes explicites pour machine learning.'''
    # drops
    df = (
        df
        .query("followers > 0 & word_count > 0")
        .drop(columns=["language", "author", "is_retweet"])
    )

    # shared_url_domain
    shared_url_split_df = df["shared_url_domain"].apply(split_url).apply(pd.Series)

    # shared_url_domain: Radical
    radical_freq_series = shared_url_split_df[0].value_counts()
    shared_url_split_df[0] = shared_url_split_df[0].map(lambda x: x if (radical_freq_series.get(x, 3) > 2) else "other")

    # shared_url_domain: Extension
    extension_freq_series = shared_url_split_df[1].value_counts()
    shared_url_split_df[1] = shared_url_split_df[1].map(lambda x: x if (extension_freq_series.get(x, 1) > 1) else "other")

    # shared_url_domain: Encoding
    df[["shared_url_radical", "shared_url_extension"]] = shared_url_split_df
    df_encoded = pd.get_dummies(df, columns=["shared_url_radical", "shared_url_extension"])

    # shared_url_count
    df_encoded["is_shared_url"] = df_encoded["shared_url_count"] > 0

    # timestamp
    timestamp_datetime_series = pd.to_datetime(df_encoded["timestamp"], unit='ms')

    df_encoded["month_post"] = timestamp_datetime_series.dt.month
    df_encoded["day_of_week_post"] = timestamp_datetime_series.dt.dayofweek
    df_encoded["hour_post"] = timestamp_datetime_series.dt.hour

    max_timestamp = df_encoded["timestamp"].max()
    df_encoded["timestamp"] = df_encoded["timestamp"].apply(lambda x: np.log10(max_timestamp + 1 - x)).replace(-np.inf, 0)

    # word_count, engagement, followers
    df_encoded["word_count"] = df_encoded["word_count"].apply(lambda x: np.log10(x+1))
    df_encoded["engagement"] = df_encoded["engagement"].apply(np.log10).replace(-np.inf, 0)
    df_encoded["followers"] = df_encoded["followers"].apply(np.log10)

    # feature1
    max_feature1 = df_encoded["feature1"].max()
    df_encoded["feature1_is_max"] = df_encoded["feature1"] == max_feature1

    # final drops
    df_encoded.drop(columns=["feature1", "shared_url_count", "shared_url_domain"], inplace=True)

    return df_encoded


def pca_transform(df, n_components):
    V_cols = [col for col in df.columns if "V" in col]

    pca = PCA(n_components=n_components)
    V_reduced = pca.fit_transform(df[V_cols])

    df_reduced = df.drop(columns=V_cols)
    df_reduced[V_cols[:n_components]] = V_reduced

    return df_reduced
