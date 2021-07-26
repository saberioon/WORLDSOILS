import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fnn import splitting_dataset
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.stats import chi2
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(prog='outlier_finder.py', description='finding outliers')
    parser.add_argument("-i", "--input", dest='input', type=str, help=" input filename", required=True)
    parser.add_argument("-o", "--output", dest='output', type=str, help="output filename ", required=True)

    return parser


def outliers_finder(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Finding and removing outliers
    :param data_frame:
    :return:
    """
    (df_X, df_y) = splitting_dataset(data_frame)
    # Define the PCA object
    pca = PCA()

    # Run PCA on scaled data and obtain the scores array
    T = pca.fit_transform(StandardScaler().fit_transform(df_X.values))

    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    robust_cov = MinCovDet().fit(T[:, :5])

    # Get the Mahalanobis distance
    m = robust_cov.mahalanobis(T[:, :5])

    data_frame['mahalanobis'] = m

    # calculate p-value for each mahalanobis distance
    data_frame['p'] = 1 - chi2.cdf(data_frame['mahalanobis'], 3)
    data_frame.sort_values('p', ascending=False)
    Drops = (data_frame['p'] <= 0.001)
    data_frame['Drops'] = (data_frame['p'] <= 0.001)

    indexNames = data_frame[data_frame['Drops'] == True].index
    print(indexNames.size)
    data_frame.drop(indexNames, inplace=True)

    return data_frame


def clean_dataset(data_frame: pd.DataFrame) -> pd.DataFrame:
    """

    :param data_frame:
    :return:
    """

    del data_frame['mahalanobis']
    del data_frame['p']
    del data_frame['Drops']

    return data_frame


def run():
    parser = parse_arg()
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df_clean = outliers_finder(df)

    df_out = clean_dataset(df_clean)
    df_out.to_csv(args.output, index=True)


if __name__ == "__main__":
    run()
    print("DONE!! Be Happy ;)")
