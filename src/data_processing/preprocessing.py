import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
import json
import re
from datetime import datetime


def add_meta_data__raw(df):
    """
    Function for adding meta data to df

    Arguments:
        * df: panda dataframe
    Outputs:
        * df: panda dataframe
    """
    time_stamp = time.time()
    # add meta data to df
    df['pk'] = df.index
    index_list = list(df.index)
    np.random.shuffle(index_list)
    df['document_id'] = index_list
    np.random.shuffle(index_list)
    df['paragraph_id'] = index_list
    df['event_time'] = time_stamp
    return df


def split_stratified_into_train_val_test(df_input,
                                         stratify_flag=True,
                                         stratify_colname='y',
                                         frac_train=0.8,
                                         frac_val=0.1, frac_test=0.1,
                                         random_state=4):
    '''
    Function for splitting a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column)

    Parameters:
        * df_input : Pandas dataframe
            Input dataframe to be split
        * stratify_flag: boolean
            Specifies if stratify is needed
        * stratify_colname : str
            The name of the column that will be used for stratification. Usually
            this column would be for the label
        * frac_train : float
        * frac_val   : float
        * frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0
        * random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split()

    Outputs:
        * data_type : Pandas dataframes
            Dataframe containing `data_type` with train, dev, test values
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            f'fractions {frac_train}, {frac_val}, {frac_test} do not add up to 1.0')

    if stratify_colname not in df_input.columns:
        raise ValueError(
            f'{stratify_colname} is not a column in the dataframe')

    X = df_input  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    if stratify_flag:
        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                              y,
                                                              stratify=y,
                                                              test_size=(
                                                                  1.0 - frac_train),
                                                              random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                          y_temp,
                                                          stratify=y_temp,
                                                          test_size=relative_frac_test,
                                                          random_state=random_state)

        assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    if not stratify_flag:
        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                              y,
                                                              test_size=(
                                                                  1.0 - frac_train),
                                                              random_state=random_state)

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                          y_temp,
                                                          test_size=relative_frac_test,
                                                          random_state=random_state)

        assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_val = df_val.copy()

    df_train['data_type'] = 'train'
    df_val['data_type'] = 'val'
    df_test['data_type'] = 'test'

    df_out = pd.concat([df_train, df_val, df_test], axis=0)

    return df_out


def make_most_common_targets(df, dependent_variables, targer_number_out):
    """
    Function for reducing the number of dependent_variables

    Arguments:
        * df: Pandas dataframe
            target datafram
        * dependent_variables: String 
            name of dependent variable column
        * targer_number_out: Intiger
            number of dependent variables to keep
    Outputs:
        * df_out: Pandas dataframe
            dataframe with additional "target" column
    """
    df_out = df.copy()
    rank_classes_df = pd.DataFrame(df[dependent_variables].value_counts())
    assert rank_classes_df.size > targer_number_out, "The number of available categories in the input df must be > targer_number_out"
    most_common_classes = rank_classes_df[0:
                                          targer_number_out-1].index.to_list()
    print(
        f"Most common {targer_number_out-1} categories idnetified: {most_common_classes}")
    cl_map = ~rank_classes_df.index.isin(most_common_classes)
    classes_to_rename = rank_classes_df.index[cl_map]
    df_out['target'] = df_out[dependent_variables]
    df_out['target'].replace(classes_to_rename, 'other', inplace=True)
    return df_out


def clean_text(words):
    '''
    Function for cleaning a text string

     Arguments:
           * word: string
               input text
           * key: string file folder_name/file_name
       Outputs:
           * text_clean: string 
               processed text
    '''
    if not words:
        return None
    remove_list = ["and", "to", "was", "it", "<<<", "into",
                   "a", "i", "with", 'that', ">>>",
                   "would", "with", "of", "[", "]",
                   "in", "he", "on", "*", "n", "at",
                   "for", "we", "y", "had", "or", "ip",
                   "from", "were", "is", "has", "her", "she",
                   "as", "this", "be", "his", "all", "my", "by", "any", "each", "will", 'a', 's', 'its', 'if', 'are']

    remove_list = remove_list + re.findall(r"<<<\[([^\]\[\r\n]*)\]>>>", words)
    words = str(words)
    words = words.lower()
    words = words.replace("<<<", "")
    words = words.replace(">>>", "")
    words = words.replace("[", "")
    words = words.replace("]", "")
    words = words.replace("(", "")
    words = words.replace(")", "")
    words = words.replace("*", "")
    words = words.replace("what's", "what is ")
    words = words.replace(".", "")
    words = words.replace(",", "")
    words = words.replace("wasn't", "was not")
    words = words.replace("can't", "cannot")
    words = words.replace("it's", "it is")
    words = words.replace("it's", "it is")
    words = words.replace("\'ve", " have ")
    words = words.replace("'s", " ")
    words = words.replace("’s", " ")
    words = words.replace("can't", "cannot ")
    words = words.replace("don't", "do not ")
    words = words.replace("doesn't", "does not ")
    words = words.replace("n't", "not ")
    words = words.replace(r"i'm", "i am")
    words = words.replace(r" m ", " am ")
    words = words.replace(r"\'re", " are ")
    words = words.replace(r"\'d", " would ")
    words = words.replace("'", "")
    words = words.replace("!", "")
    words = words.replace(r"\'ll", " will ")
    words = words.replace(r"´", "")
    words = words.replace("`", "")
    words = words.replace("\r\n", " ")
    words = words.replace("the", "")
    split_out = [x for x in words.split() if x not in remove_list]
    text_clean = " ".join(split_out)
    return text_clean


def down_sample(df):
    """
    Function for Downsample `other`

    Arguments:
        Inputs: 
            * df: panda dataframe
                input df
       Outputs:
           * df_out: panda dataframe 
               updated df
    """
    df2 = df[df.target == 'other'].copy()
    df1 = df[df.target != 'other'].copy()
    df2 = df2.sample(
        n=int(np.mean(df1['target'].value_counts())), random_state=1).copy()
    df_out = df1.append(df2).copy()
    df_out.sample(n=df_out.shape[0], random_state=1).reset_index(
        drop=True, inplace=True)

    return df_out


def get_spacy_embeddings(df, column_name, model):
    """
    Function to calculate text embedding

    Arguments:
        * df: Panda Dataframe
            input df
        * column_name: string
            column name with text
       Outputs:
           * df: Panda Dataframe
               updatated df with a `feature` column 
    """
    df['features'] = df[column_name].apply(
        lambda x: json.dumps((model(x).vector.tolist())))
    return df


def data_process_main(input_data_path):
    nlp = spacy.load("en_core_web_md")
    df = pd.read_parquet(input_data_path)
    # adding meta data to df
    df = add_meta_data__raw(df)
    # making train test split
    df_out = split_stratified_into_train_val_test(df_input=df,
                                                  stratify_flag=True,
                                                  stratify_colname='clause_type',
                                                  frac_train=0.8,
                                                  frac_val=0.1, frac_test=0.1,
                                                  random_state=4)
    # reducing the number of dependent variable
    df_out = make_most_common_targets(df_out, 'clause_type', 6)
    # cleaning the text
    df_out['clean_description'] = df_out['text'].apply(clean_text)
    # selecting the output columns
    print(df.columns)
    df_out_test = df_out[['pk', 'document_id', 'paragraph_id', 'text',
                          'clean_description', 'data_type', 'clause_type', 'target', 'event_time']].copy()
    df_out_test = down_sample(df_out_test)
    df_out_test = get_spacy_embeddings(df_out_test, 'clean_description', nlp)

    return df_out_test


if __name__ == "__main__":
    filtered_data = data_process_main(
        '/opt/ml/processing/input/data/ledgar.parquet.gzip')
    filtered_data.to_parquet('/opt/ml/processing/output/df.parquet.gzip')
