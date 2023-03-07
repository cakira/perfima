#!/usr/bin/env python

import gspread
import pandas as pd
import re
from pathlib import Path
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier


def main():
    nb = read_nubank('inbox/nubank-2023-01.csv')
    nb2 = read_nubank('inbox/nubank-2023-02.csv')
    bb = read_bb('inbox/bb-2023-01.csv')
    al = read_alelo('inbox/alelo-2023-01.txt')
    prev = read_gsheet('Contas Akira', '2022-11')
    prev2 = read_gsheet('Contas Akira', '2022-12')

    prev = pd.concat([prev, prev2])
    total_unfiltered = pd.concat([nb, nb2, bb, al]).fillna('')
    total = date_filter(total_unfiltered, '2023-01-1', '2023-02-1')

    x_treino, x_teste, y_treino, y_teste, le_cat = preprocess_and_split(prev, total)

    knn = KNeighborsClassifier(n_neighbors=28)
    total['category'] = classify(knn, x_treino, x_teste, y_treino, le_cat)

    dt = DecisionTreeClassifier()
    total['category2'] = classify(dt, x_treino, x_teste, y_treino, le_cat)

    write_gsheet(total, 'Contas Akira', '2023-01-raw')


def read_nubank(filename: Path):
    df = pd.read_csv(filename)
    df = df.rename({'category':'original_category', 'title':'description', 'amount':'value'}, axis='columns')
    df['source'] = 'nubank'
    df['value'] = -df['value']
    df['category'] = ''
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'category', 'description', 'value', 'source', 'original_category']]
    return df


def read_bb(filename: Path):
    bb=pd.read_csv(filename, encoding='cp1252')
    bb = bb.rename({'Data':'date', 'Histórico':'description', 'Valor':'value'}, axis='columns')
    bb['source'] = 'BB'
    bb['date'] = pd.to_datetime(bb['date'], dayfirst=True)
    bb = bb[~ ((bb['description'] == 'BB Rende Fácil') & bb['Data do Balancete'].isnull())]
    bb = bb[['date', 'description', 'value', 'source', 'Dependencia Origem', 'Data do Balancete', 'Número do documento']]
    bb.drop(bb[bb['description'] == 'S A L D O'].index, inplace=True)
    return bb


def read_alelo(filename: Path):
    with open(filename) as alelo_txt:
        lines = list(alelo_txt)
    entry_match = re.compile(r'(.*)([0-9]{2}/[0-9]{2}).*\n.* ([0-9.]+,[0-9]{2})$')
    dates = []
    descriptions = []
    values = []
    for i in range(len(lines)-1):
        dual_line = lines[i]+lines[i+1]
        ma = entry_match.search(dual_line)
        if ma is not None:
            full_date = ma.group(2) + ('/2022' if ma.group(2).endswith('/12') else '/2023')
            dates.append(pd.to_datetime(full_date, dayfirst=True))
            descriptions.append(ma.group(1))
            value = -float(ma.group(3).replace('.', '').replace(',', '.'))
            if 'Benefício' in ma.group(1):
                value = -value
            values.append(value)
    al=pd.DataFrame(data={'date':dates, 'description':descriptions, 'value':values})
    al['source'] = 'Alelo'
    return al


def read_gsheet(doc_name: str, sheet_name: str):
    gspread_connector = gspread.service_account()
    spreadsheet = gspread_connector.open(doc_name)
    prev_worksheet = spreadsheet.worksheet(sheet_name)
    prev = pd.DataFrame(prev_worksheet.get_all_records())
    prev = prev.rename({'Data':'textdate',
                        'Categoria':'category',
                        'Nome':'description',
                        'Valor':'value',
                        'Fonte':'source',
                        'Categoria Original':'original_category'}, axis='columns')
    prev['date'] = pd.to_datetime(prev['textdate'].str[:5] + '/2022', dayfirst=True)
    prev.drop(['textdate'], axis=1, inplace=True)
    prev['value'] = prev['value'].apply(lambda x: str(x).replace('R$ ','').replace('.','').replace(',','.')).astype('float')
    return prev


def date_filter(total_unfiltered, start_date: str, end_date: str):
    total_unfiltered.insert(0, 'order', range(len(total_unfiltered)))
    total_unfiltered = total_unfiltered.sort_values(by=['date', 'order']).drop(['order'], axis=1)
    total = total_unfiltered[(total_unfiltered['date']>=pd.to_datetime(start_date))
                            & (total_unfiltered['date']<pd.to_datetime(end_date))]
    total = total.reset_index().drop(['index'], axis=1)
    return total


def preprocess_and_split(prev, total):
    prevf = pd.concat([prev, total]).fillna('')

    prevf['wd'] = prevf['date'].dt.dayofweek
    prevf['md'] = prevf['date'].dt.day
    prevf['month'] = prevf['date'].dt.month

    # print(prevf['original_category'])
    le_orcat = preprocessing.LabelEncoder()
    le_orcat.fit(prevf['original_category'].values)
    prevf['orcatf'] = le_orcat.transform(prevf['original_category'].values)

    le_cat = preprocessing.LabelEncoder()
    le_cat.fit(prevf['category'].values)
    prevf['catf'] = le_cat.transform(prevf['category'].values)

    le_desc = preprocessing.LabelEncoder()
    le_desc.fit(prevf['description'].values)
    prevf['descf'] = le_desc.transform(prevf['description'].values)

    le_src = preprocessing.LabelEncoder()
    le_desc.fit(prevf['source'].values)
    prevf['sourcef'] = le_desc.transform(prevf['source'].values)

    prevf.drop(['date', 'category', 'description', 'source', 'Comentário'], axis=1, inplace=True)
    prevf.drop(['original_category', 'Dependencia Origem', 'Data do Balancete', 'Número do documento'], axis=1, inplace=True)


    Y = prevf['catf'].values
    prevf.drop('catf', axis=1, inplace=True)

    colunas            = prevf.columns
    norm               = preprocessing.Normalizer()
    dados_normalizados = norm.transform(prevf.values)
    df_norm            = pd.DataFrame(dados_normalizados, columns=colunas)
    X                  = dados_normalizados

    x_treino = X[0:len(prev)]
    y_treino = Y[0:len(prev)]
    x_teste = X[len(prev):]
    y_teste = Y[len(prev):]
    return x_treino, x_teste, y_treino, y_teste, le_cat


def write_gsheet(total, doc_name: str, sheet_name: str) -> None:
    gspread_connector = gspread.service_account()
    spreadsheet = gspread_connector.open(doc_name)
    worksheet_list = [ws.title for ws in spreadsheet.worksheets()]
    if sheet_name in worksheet_list:
        worksheet = spreadsheet.worksheet(sheet_name)
    else:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(total.index) + 1, cols=7)
    worksheet.update('A1', [['Data', 'Categoria', 'Nome', 'Valor', 'Fonte', 'Categoria Original', 'Comentário']])
    total['date'] = total['date'].astype(str)
    worksheet.update('A2', total.values.tolist(), value_input_option='USER_ENTERED');


def classify(method, x_treino, x_teste, y_treino, le_cat):
    method.fit(x_treino, y_treino)
    method.score(x_treino, y_treino)
    pred_knn = method.predict(x_teste)
    return le_cat.inverse_transform(pred_knn)


if __name__ == '__main__':
    main()
