import datetime
import os
import pickle
import string
import sys
import time
import zlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse

from nltk.corpus import stopwords
from nltk import sent_tokenize, regexp_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2

from sqlalchemy import select
from tqdm import tqdm
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

from settings.redis import get_redis
from settings.config import *

import logging
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


pd.options.mode.chained_assignment = None
tqdm.pandas()

data_path = DATA_PATH

class DataProcessor:
    def __init__(self, df_data, df_abbr, key):
        self._df_data = df_data
        self.__df_abbr = df_abbr
        self.key = key
        logger.info('Initialized Data Processor')

    def __replace_descrition(self):
        logger.info('Replacing description')
        # chars = '!"\\#\\$%\\&\'\\(\\)\\*\\+,\\-\\./:;<=>\\?@\\[\\\\\\]\\^`\\{\\|\\}\\~'
        chars = re.escape(string.punctuation)

        self._df_data['description'] = self._df_data['description'].astype(str).str.replace('_x000D_', '')
        self._df_data['description'] = self._df_data['description'].str.replace('\n', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('\t', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Заводские данные о товаре', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Серия', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Модель', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Артикул производителя', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Бренд', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Код товара', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Основные характеристики', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Страна-производитель', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Тип', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Пол', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Размер RU', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Размер производителя', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Сезон', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Материалы и цвет', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Материал стельки', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Цвет', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Цвет производителя', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Узор', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Дополнительная информация', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Длина стопы, см', ' ')
        self._df_data['description'] = self._df_data['description'].str.replace('Для занятий спортом', ' ')
        self._df_data['description'] = self._df_data['description'].str.lower()
        self._df_data['description'] = self._df_data['description'].str.replace(r'\s+', ' ', regex=True)

        self._df_data = self._df_data.dropna(subset=['description']).reset_index(drop=True)
        logger.info('Finished replacing description')

    def __null_cleaner(self):
        logger.info('Cleaning data')
        self._df_data = self._df_data[self._df_data['description'].notnull()]
        self.__df_abbr = self.__df_abbr[self.__df_abbr['abbr_short'].notnull()]
        self.__df_abbr = self.__df_abbr[self.__df_abbr['abbr_full'].notnull()]
        self._df_data = self._df_data[self._df_data['id'].notnull()]
        logger.info('Finished cleaning data')

    def __replacing_abbrs(self):
        logger.info('Replacing abbreviations')
        for row_abbr in tqdm(self.__df_abbr.itertuples(), total=self.__df_abbr.shape[0]):
            self._df_data['description'] = self._df_data['description'].str.replace(r'\b{}\b'.format(row_abbr.abbr_short), row_abbr.abbr_full, regex=True)
        logger.info('Finished replacing abbreviations')

    @lru_cache(maxsize=None)
    def __normalize_one_token(self, morph, word: str) -> str:
        """
        Нормализация одного слова
        """
        return (morph.parse(word)[0].normal_form)

    def __tokenize_n_lemmatize(self, text, morph, stop_words=None, normalize: bool = True,
        regexp: str = r'(?u)\b\w{4,}\b',
        min_length: int = 4) -> list:
        """
        Функция вызывает функцию нормализации и функцию удаления стоп-слов
        """
        words = [w for sent in sent_tokenize(text) for w in regexp_tokenize(sent, regexp)]
        if normalize:
            words = [self.__normalize_one_token(morph, word) for word in words]
        if stopwords:
            stop_words = set(stop_words)
            words = [word for word in words if word not in stop_words and len(word) >= min_length]
        return words

    def __tokenize_lemmatize(self):
        logger.info('Tokenizing and lemmatizing')
        morph = pymorphy2.MorphAnalyzer()
        stop_r = stopwords.words('russian')
        proc_data = self._df_data[['id', 'description', 'url']]
        proc_data.description = proc_data.description.progress_apply(lambda x: ' '.join(
            self.__tokenize_n_lemmatize(text=x, morph=morph, stop_words=stop_r)))
        proc_data.reset_index(drop=True, inplace=True)
        logger.info('Finished tokenizing and lemmatizing')
        return proc_data

    def full_processing(self):
        logger.info('Begin data processing')
        self.__null_cleaner()
        self.__replacing_abbrs()
        self.__replace_descrition()
        proc_data = self.__tokenize_lemmatize()
        logger.info('Finisehd data processing')
        return proc_data

class TrainProc:

    def __init__(self, proc_data: pd.DataFrame, abbr: pd.DataFrame, key):
        logger.info('Initializing Training class')
        self.__df_abbr = abbr
        self._proc_data = proc_data
        self.__redis = get_redis()
        self.key = key

    def _save_train_data(self, save_redis=True, save_file=True, **kwargs):
        if save_redis:
            start = time.time()
            logger.debug('Saving model data to Redis')
            for kw in kwargs.keys():
                s = zlib.compress(pickle.dumps(kwargs[kw]))
                self.__redis.set(kw, s)
            logger.debug(f'Successfully saved model data to Redis. Taken time: {time.time() - start}')
        if save_file:
            start = time.time()
            for kw in kwargs.keys():
                logger.debug(f"Saving model data to Picke file at {DATA_PATH}/{kw}_model.pkl")
                pickle.dump(kwargs[kw], open(f"{DATA_PATH}/{kw}_model.pkl", 'wb'))
            logger.debug(f'Successfully saved model data to Picke file. Taken time: {time.time() - start}')

    def train(self, save_redis=True, save_file=True):
        logger.debug('Begin fitting model')
        stop_r = stopwords.words('russian')
        idf_vector = TfidfVectorizer(stop_words=stop_r, min_df=0.00005)
        proc_data_vector = idf_vector.fit_transform(self._proc_data.description)
        logger.debug('Calclulated TF-IDF vectors. Fitting completed.')
        data = dict(proc_data=self._proc_data, proc_data_vector=proc_data_vector, vector=idf_vector)
        if save_redis or save_file:
            save_data = {self.key: data}
            try:
                self._save_train_data(save_redis, save_file, **save_data)
            except Exception as e:
                logger.critical(f'Data is not saved due to {str(e)}')
        return data


class PredictProc:

    def __init__(self, abbr, key, model_data=None):
        logger.debug('Initializing Predictor model')
        self.learning = False
        self.key = key
        self.__df_abbr = abbr
        self.__redis = get_redis()
        try:
            self._train_data = model_data[key]
        except:
            logger.error('No model data inserted, trying to load')

    def _save_train_data(self, save_redis=True, save_file=True, **kwargs):
        if save_redis:
            start = time.time()
            logger.debug('Saving model data to Redis')
            for kw in kwargs.keys():
                s = zlib.compress(pickle.dumps(kwargs[kw]))
                self.__redis.set(kw, s)
            logger.debug(f'Successfully saved model data to Redis. Taken time: {time.time() - start}')
        if save_file:
            start = time.time()
            for kw in kwargs.keys():
                logger.debug(f"Saving model data to Picke file at {DATA_PATH}/{kw}_model.pkl")
                pickle.dump(kwargs[kw], open(f"{DATA_PATH}/{kw}_model.pkl", 'wb'))
            logger.debug(f'Successfully saved model data to Picke file. Taken time: {time.time() - start}')

    def _load_train_data(self, *args, from_file=False):
        train_data = {}
        if from_file:
            for name in args:
                try:
                    logger.debug(f"Loading Picke file at {DATA_PATH}/{name}_model.pkl")
                    train_data = pickle.load(open(f"{DATA_PATH}/{name}_model.pkl", 'rb'))
                except Exception as e:
                    logger.critical(f"Could not load Picke model at {DATA_PATH}/{name}_model.pkl" + str(e))
                    raise Exception(f'No saved pkl model found!')
        else:
            logger.debug("Loading data from Redis")
            try:
                for name in args:
                    train_data = pickle.loads(zlib.decompress(self.__redis.get(name)))
            except Exception as e:
                logger.critical('Could not load model from Redis ' + str(e))
                raise Exception(f'No data with keys {args} in Redis')
        return train_data

    def predict(self, inc_ids:tuple = None, description=None, from_file=False, reload_model=False) -> dict:
        """
        Функция получает на вход id инцидента или его описание и выдает похожие инциденты
        При этом если на входе id инцидента, топ 1 отрезаем - это он сам и будет
        """
        start = time.time()
        if hasattr(self, '_train_data') and not reload_model:
            logger.debug('Model for prediction - RAM')
        else:
            if from_file:
                logger.debug('Model for prediction - Pickle')
                self._train_data = self._load_train_data(self.key, from_file=True)
            else:
                logger.debug('Model for prediction - Redis')
                self._train_data = self._load_train_data(self.key, from_file=False)

        proc_data = self._train_data['proc_data']
        proc_data_vector = self._train_data['proc_data_vector']
        vectorizer = self._train_data['vector']

        print(time.time() - start)


        if proc_data is None or proc_data_vector is None or vectorizer is None:
            logger.critical('ERROR - could not load moder for prediction')
            raise Exception('No loaded model present, load or fit first')

        if description:
            logger.debug(f"Start predicting for {description}")
            test_set = pd.DataFrame({'description': [description], 'id': [0], 'url':['none']})
            test_set = DataProcessor(df_data=test_set, df_abbr=self.__df_abbr, key=self.key).full_processing()
            logger.debug(f"The processing for description done, got {test_set.description}")

        logger.info("Test set is created based on IDs or description")

        logger.info("Applying TF-IDF model to test set")
        train_set = proc_data.copy()
        train_set_vector = proc_data_vector
        test_set_vector = vectorizer.transform(test_set.description)

        logger.info("Calculating cosine similarity")
        cosSim = cosine_similarity(train_set_vector, test_set_vector)
        cosSim = (np.rint(cosSim * 100))
        answer = {}

        if description:
            logger.debug(f"Returning data for description: {description}")
            train_set['Prob'] = cosSim
        return train_set.sort_values(by=['Prob'], ascending=False)[['id', 'Prob', 'url']].head(5).to_dict('records')



def load_excel_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f'Loading data from Excel table at: {data_path}')
        df_data = pd.read_excel(io=data_path, usecols="A:I", engine='openpyxl')
        logger.info(f'Successfully loaded data from Excel table at: {data_path}')
        return df_data
    except Exception as e:
        logger.critical(f'Could not load data from from Excel table at: {data_path}: {e}')
        raise e

def load_csv_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f'Loading abbreviations data from csv file at: {data_path}')
        df_abbr = pd.read_csv(data_path, sep=';')
        print('loaded abbrs')
        return df_abbr
    except Exception as e:
        logger.critical(f'Could not load abbreviations data from csv file at: {data_path}: {e}')
        raise e


#id categ_root categ1 categ2 categ3 properties categ4 name url
def fresh_train(keys:tuple, excel:bool=True, path=data_path):
    model_data = {}
    logger.info('Running fresh train function')
    if excel:
        for dir in [f for f in Path(path).iterdir() if f.is_dir()]:
            key = dir.name
            if key in keys:
                logger.info(f'Fresh train is using Excel {key} data')
                df_data = load_excel_data(data_path = str(dir)+f"/{key}.xlsx")
                df_data['description'] = df_data['categ3'] + df_data['properties'] + df_data['name']
                df_abbr = load_csv_data(data_path = str(dir)+f"/{key}_abbr.csv")
                data_processor = DataProcessor(df_data=df_data, df_abbr=df_abbr, key=key)

                proc_data = data_processor.full_processing()
                train = TrainProc(proc_data=proc_data, abbr=df_abbr, key=key)
                model_data[key] = train.train(save_redis=False, save_file=True)
    # else
    # logger.info('Fresh train is using SQL - Oracle (SM) and Postgre (abbreviations)')
    # query_abbr = select(Abbr.abbr_short, Abbr.abbr_full)
    # df_abbr =pd.read_sql(query_abbr, sync_engine)
    # for key in keys:
    # path = 'src/apps/analyzer/sql/query.sql'
    # groups = key_to_group[key]
    # df_data = query_and_parse_oracle(conn, path, groups)
    # df_data['created'] = df_data['created'].astype(str)
    # data_processor = DataProcessor(df_data=df_data, df_abbr=df_abbr, key=key)
    # proc_data = data_processor.full_processing()
    # train = TrainProc(proc_data=proc_data, abbr=df_abbr, key=key)
    # model_data[key] = train.train(save_redis=True, save_file=True)
    return model_data

def use_predictor(keys:tuple, model_data, excel=True, path=data_path):
    predict = {}
    if excel:
        for dir in [f for f in Path(path).iterdir() if f.is_dir()]:
            key = dir.name
            if key in keys:
                logger.info(f'Using {key} csv data for abbreviatons')
                df_abbr = load_csv_data(data_path= str(dir)+f"/{key}_abbr.csv")
                predict[key] = PredictProc(abbr=df_abbr, key=key, model_data=model_data)
    # else:
    # logger.info('Using Postges data for abbreviatons')
    # query_abbr = select(Abbr.abbr_short, Abbr.abbr_full)
    # df_abbr = pd.read_sql(query_abbr, sync_engine)
    # for key in keys:
    # predict[key] = PredictProc(abbr=df_abbr, key=key, model_data = model_data)
    #
    # logger.info(f'Returning dict of predictors for {keys}')
    return predict

model_data = fresh_train(keys=KEYS, excel=EXCEL_LEARN, path=DATA_PATH)
predict = use_predictor( model_data=model_data, keys=KEYS, excel=True, path=DATA_PATH)
########################################################
