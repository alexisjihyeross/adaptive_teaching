import pandas as pd
import re
import urllib.request
from tqdm import tqdm

from src.dataset import Dataset
from src.verbs.verb_helpers import *
from pyinflect import getInflection


def download_unimorph_data():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/unimorph/eng/master/eng.segmentations",
        "eng.segmentations",
    )


def load_unimorph_data():
    """Loads and processes unimorph data"""
    download_unimorph_data()

    def get_ending(row):
        try:
            lemma, ending = row["segmentations"].split("|")
            assert lemma == row["lemma"]
        except:
            return "-"
        return ending

    def is_past(attr):
        # filter PTCP so that eat -> eaten not included
        return ("PST" in attr) and ("PTCP" not in attr)

    tqdm.pandas()
    df = pd.read_csv(
        "eng.segmentations",
        sep="\t",
        header=None,
        names=["lemma", "form", "type", "segmentations"],
    )
    df = df.dropna()
    df["attributes"] = df.progress_apply(lambda row: row["type"].split("|")[-1], axis=1)
    df["is_verb"] = df.progress_apply(
        lambda row: row["attributes"].startswith("V"), axis=1
    )
    df = df.drop_duplicates(subset=["lemma", "form"])

    # this rules out cases like thwonk where the type is V|PST
    verb_df = df
    verb_df["ending"] = verb_df.apply(lambda row: get_ending(row), axis=1)

    # TODO: do more complex matching? sometimes just has PST, sometimes has V.PST?
    verb_df["is_past"] = verb_df.apply(lambda row: is_past(row["attributes"]), axis=1)
    past_df = verb_df[verb_df["is_past"]]

    # Categorize verbs
    processed_df = past_df
    processed_df["category"] = past_df.apply(categorize_verb_wrapper, axis=1)

    return processed_df


def load_dataset():
    print("Loading dataset...")
    df = load_unimorph_data()

    df = df[df["category"].isin(["+d", "+ed", "y_to_ied", "+consonant+ed"])]

    inputs = list(df["lemma"].values)
    outputs = list(df["category"].values)

    dataset = VerbsDataset(inputs, outputs)
    print("Done.")
    return dataset


def categorize_verb_wrapper(row):
    lemma, form = row["lemma"], row["form"]
    return categorize_verb(lemma, form)


def categorize_verb(lemma, form):
    for category, condition in VERB_CATEGORIES.items():
        if condition(lemma, form):
            return category
    return "unknown"


# used when there is an error in computing the Verb Label
class VerbCategoryError(Exception):
    pass


def get_verb_category(lemma, dataset):
    """
    Helper function to get the past tense form of a verb. First checks to see if the lemma is in the dataset. If not, uses spacy/pyinflect.
    """

    try:
        return dataset.get_label(lemma)

    except KeyError:
        print(f"lemma {lemma} does not exist in the dataset. trying to inflect it")
        try:
            form = getInflection(lemma, tag="VBD")[0]
        except Exception as e:
            print("Error getting form:", e)
            raise VerbCategoryError()

        cat = categorize_verb(lemma, form)
        return cat


class VerbsDataset(Dataset):
    def __init__(self, *args):
        super().__init__(*args)

    def check_input_validity(self, inp):
        # TODO: check if in some lexicon?
        return True

    def check_output_validity(self, out):
        # classification, so can just check if in label space
        return out in self.unique_outputs
