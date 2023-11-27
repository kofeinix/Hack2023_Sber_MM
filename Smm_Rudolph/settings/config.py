import os
EXCEL_LEARN = True
EXCEL_PREDICT = True
LOAD_PICKLE = True
NOT_USE_RAM_MODEL = False

BASE_DIR = os.environ.get('BASE_DIR')
DATA_PATH = str(BASE_DIR) + 'analyzer_data/'

KEYS = ('smm',)
