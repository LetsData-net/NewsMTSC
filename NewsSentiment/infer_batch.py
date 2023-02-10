from timeit import default_timer
from typing import List, Union, Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import re
from NewsSentiment import TargetSentimentClassifier
from NewsSentiment.dataset import FXEasyTokenizer
from NewsSentiment.customexceptions import TooLongTextException
from tqdm import tqdm


class BatchTFC():
    def __init__(self, data: pd.DataFrame = None, text_col: str = None, start_ind_col: str = None,
                 end_ind_col: str = None, name_col: str = None, true_name_col: str = None):
        self.tsc = TargetSentimentClassifier()
        self.__dataframe = data
        self.bad_examples = pd.DataFrame()
        self.add_temp_entity_id()
        self.batches = None
        self.processed_data = None
        self.batch_size = None
        self.prepared_data = None
        self.text_col = text_col
        self.start_ind_col = start_ind_col
        self.end_ind_col = end_ind_col
        self.name_col = name_col
        self.true_name_col = true_name_col
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @property
    def dataframe(self):
        df = pd.concat([self.__dataframe, self.bad_examples], ignore_index=False)
        return df

    @dataframe.setter
    def dataframe(self, in_dataframe):
        self.__dataframe = in_dataframe

    def add_temp_entity_id(self):
        """Add temporary entity id to append results back to original data
        """
        self.__dataframe['temp_entity_id'] = self.__dataframe.index

    def subset_by_pattern(self, pattern: Union[List[str], str]):
        dataframe = self.__dataframe

        if self.true_name_col is not None:
            target_col = self.true_name_col
        elif self.name_col is not None:
            target_col = self.name_col
        else:
            raise ValueError('Provide target column name before subsetting')

        if isinstance(pattern, list):
            dataframe = dataframe[dataframe[target_col].isin(pattern)].reset_index(drop=True)
        else:
            dataframe = dataframe[dataframe[target_col].str.contains(pattern, case=False)].reset_index(drop=True)
        self.__dataframe = dataframe

    def prep_record(self, text: str, entity_start_ind: int, entity_end_ind: int):
        left = FXEasyTokenizer.prepare_left_segment(text[:entity_start_ind])
        target = FXEasyTokenizer.prepare_target_mention(text[entity_start_ind:entity_end_ind])
        right = FXEasyTokenizer.prepare_right_segment(text[entity_end_ind:])

        indexed_example = self.tsc.tokenizer.create_model_input_seqs(
            left, target, right, []
        )
        return indexed_example

    def prep_batches(self, batch_size, parallel: bool = False):

        if parallel:
            pass
        else:
            prepared_data = []
            for _, record in tqdm(self.__dataframe.iterrows(), total=self.__dataframe.shape[0]):
                try:
                    indexed_record = self.prep_record(
                        record[self.text_col],
                        record[self.start_ind_col],
                        record[self.end_ind_col])
                    prepared_data.append(indexed_record)
                except TooLongTextException:
                    self.bad_examples = self.bad_examples.append(record.to_dict(), ignore_index=True)
                    self.__dataframe = self.__dataframe.drop(labels=_, axis=0)

        self.batches = DataLoader(dataset=prepared_data, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.__dataframe = self.__dataframe.reset_index(drop=True)

    def infer(self):
        res = []
        prev_batch_n = 0
        batches = self.batches
        dataframe = self.__dataframe

        for batch in tqdm(batches):
            inputs = self.tsc.instructor.select_inputs(batch)
            iter_outputs = self.tsc.model(inputs)

            iter_res = []
            for n, outputs in enumerate(iter_outputs):
                class_probabilites = F.softmax(outputs, dim=-1).reshape((3,)).to(self.device).tolist()

                classification_result = []
                for class_id, class_prob in enumerate(class_probabilites):
                    classification_result.append(
                        {
                            "sentiment": self.tsc.polarities_inverse[class_id].title(),
                            "sentiment_prob": class_prob
                        }
                    )
                    highest_prob_sentiment = max(classification_result, key=lambda x: x['sentiment_prob'])
                    highest_prob_sentiment['temp_entity_id'] = dataframe.loc[n+prev_batch_n, 'temp_entity_id']

                iter_res.append(highest_prob_sentiment)
            prev_batch_n += self.batch_size
            res.extend(iter_res)

        processed_data = pd.DataFrame.from_dict(res)

        self.dataframe = pd.merge(dataframe, processed_data, how='left', on='temp_entity_id')

