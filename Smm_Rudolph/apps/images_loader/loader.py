import asyncio
import io
from io import BytesIO

import aiohttp
import pandas as pd
import requests
from PIL import Image
from sqlalchemy import select
import urllib3
#from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
from settings.database import session,engine, Images
import os
from tqdm.asyncio import trange, tqdm
urllib3.disable_warnings()
import time

start_time = time.time()
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

async def save_new_data():
    statement = select(Images.id, Images.url)
    result = session.execute(statement)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    existing_files = [f.split('.')[0] for f in os.listdir(f'files/market/')]
    df = df[~df['id'].isin(existing_files)]
    async with aiohttp.ClientSession() as requst_session:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            async with requst_session.get(row.url, verify_ssl=False) as resp:
                content = await resp.read()
                try:
                    im = Image.open(BytesIO(content))
                    im = im.convert('RGB')
                    im.save(f'files/market/{row.id}.jpg')
                except:
                    print(f'error with {row.url}')

asyncio.run(save_new_data())
print("--- %s seconds ---" % (time.time() - start_time))