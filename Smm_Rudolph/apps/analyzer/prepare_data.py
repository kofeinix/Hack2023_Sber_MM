import pandas as pd

from settings.config import DATA_PATH
from settings.database import Pages, session

def to_dict(row):
    if row is None:
        return None
    rtn_dict = dict()
    keys = row.__table__.columns.keys()
    for key in keys:
        rtn_dict[key] = getattr(row, key)
    return rtn_dict

def exportexcel(key):
    data = session.query(Pages).all()
    data_list = [to_dict(item) for item in data]
    df = pd.DataFrame(data_list)
    filename = DATA_PATH+f'{key}/{key}.xlsx'
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, sheet_name='data', index=False)
    writer.close()
    print(f'saved as {filename}')

exportexcel('smm')