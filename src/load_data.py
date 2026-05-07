import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='latin-1')


    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    df = load_data("data/spam.csv")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
