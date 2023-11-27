import json

import g4f
import asyncio
from tqdm.asyncio import tqdm
import pandas as pd

from settings.database import session, Pages

texts = ['привет', 'как ты']
async def run_provider(text:str):
    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": f"Сократи текст: {text}"}],
        )
        return response
    except Exception as e:
        print(e)

async def run_all():
    query = session.query(Pages.id, Pages.jsfiled)
    df = pd.read_sql(query.statement, session.bind)
    for index, row in df.iterrows():
        jsfiled = json.loads(row['jsfiled'])
        jsfiled_parsed = []
        for key in ['Модель', 'Артикул производителя', 'Бренд', 'Код товара', 'Страна-производитель',
                    'Тип', 'Пол', 'Сезон', 'Материал верха', 'Цвет']:
            try:
                jsfiled_parsed.append(f'{key}: {str(jsfiled[key])}')
            except:
                pass
        jsfiled_parsed = ', '.join(jsfiled_parsed)
        print(jsfiled_parsed)
        print('sending one bit')
        response = await run_provider(jsfiled_parsed)
        print(response)

asyncio.run(run_all())
