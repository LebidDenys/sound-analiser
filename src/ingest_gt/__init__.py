
import pandas as pd

from src.io_manager import load_json


def load_gt(path):
    """ Load JSON with the ground truths

    Args:
        path (Union[str, bytes, os.PathLike]): path to the JSON file with the annotation results

    Returns:
        pd.DataFrame: data reformatted into DataFrame

    """
    gt = load_json(path)

    return pd.DataFrame(gt[0]['segmentations'])


def create_gt(gt_df, X):
    """Take annotations from the JSON, data analysis result, and produce labels.

    1 - was gunshot/explosion
    0 - no gunshot/explosion

    Args:
        gt_df (pd.DataFrame): DataFrame from the JSON annotation file
        X (pd.DataFrame): DataFrame with the stats performed on each chunk of the audio

    Returns:
        pd.Series: Series with appropriate labels

    """
    # ret = pd.Series(index=X.index)

    temp = X[['start_time', 'end_time']].copy()
    temp['label'] = temp.apply(
        lambda row: ((row['start_time'] <= gt_df['end_time']) & (row['end_time'] >= gt_df['start_time'])).any(), axis=1).astype(bool)

    return temp['label'].astype(int)
