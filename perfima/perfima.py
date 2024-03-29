#!/usr/bin/env python

import re
from pathlib import Path

import gspread
import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

SPREADSHEET_NAME = 'Contas Akira'
YEAR = 2023
MONTH = 3


def main():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    next_month_with_year = f'{YEAR}-{MONTH+1:02}' if MONTH < 12 else f'{YEAR+1}-01'

    nb = read_nubank(f'inbox/nubank-{YEAR}-{MONTH:02}.csv')
    nb2 = read_nubank(f'inbox/nubank-{next_month_with_year}.csv')
    bb = read_bb(f'inbox/bb-{YEAR}-{MONTH:02}.csv')
    al = read_alelo(f'inbox/alelo-{YEAR}-{MONTH:02}.txt')
    current_entries_unfiltered = pd.concat([nb, nb2, bb, al]).fillna('')
    current_entries = date_filter(current_entries_unfiltered,
                                  f'{YEAR}-{MONTH:02}-1',
                                  f'{next_month_with_year}-1')

    previous_entries = get_previous_entries(SPREADSHEET_NAME)

    x_treino, x_teste, y_treino, y_teste, le_cat = preprocess_and_split(
        previous_entries, current_entries)

    knn = KNeighborsClassifier(n_neighbors=28)
    current_entries['category'] = classify(knn, x_treino, x_teste, y_treino,
                                           le_cat)

    dt = DecisionTreeClassifier()
    current_entries['category2'] = classify(dt, x_treino, x_teste, y_treino,
                                            le_cat)

    write_gsheet(current_entries, SPREADSHEET_NAME, f'{YEAR}-{MONTH:02}-raw')


def read_nubank(filename: Path):
    df = pd.read_csv(filename)
    df = df.rename(
        {
            'category': 'original_category',
            'title': 'description',
            'amount': 'value'
        },
        axis='columns')
    df['source'] = 'nubank'
    df['value'] = -df['value']
    df['category'] = ''
    df['date'] = pd.to_datetime(df['date'])
    df = df[[
        'date', 'category', 'description', 'value', 'source',
        'original_category'
    ]]
    return df


def read_bb(filename: Path):
    bb = pd.read_csv(filename, encoding='cp1252')
    bb = bb.rename(
        {
            'Data': 'date',
            'Histórico': 'description',
            'Valor': 'value'
        },
        axis='columns')
    bb['source'] = 'BB'
    bb['date'] = pd.to_datetime(bb['date'], dayfirst=True)
    bb = bb[~((bb['description'] == 'BB Rende Fácil')
              & bb['Data do Balancete'].isnull())]
    bb = bb[[
        'date', 'description', 'value', 'source', 'Dependencia Origem',
        'Data do Balancete', 'Número do documento'
    ]]
    bb.drop(bb[bb['description'] == 'S A L D O'].index, inplace=True)
    return bb


def read_alelo(filename: Path):
    with open(filename) as alelo_txt:
        lines = list(alelo_txt)
    entry_match = re.compile(
        r'(.*)([0-9]{2}/[0-9]{2}).*\n.* ([0-9.]+,[0-9]{2})$')
    dates = []
    descriptions = []
    values = []
    for i in range(len(lines) - 1):
        dual_line = lines[i] + lines[i + 1]
        ma = entry_match.search(dual_line)
        if ma is not None:
            full_date = ma.group(2) + (
                f'/{YEAR-1}' if ma.group(2).endswith('/12') else f'/{YEAR}')
            dates.append(pd.to_datetime(full_date, dayfirst=True))
            descriptions.append(ma.group(1))
            value = -float(ma.group(3).replace('.', '').replace(',', '.'))
            if 'Benefício' in ma.group(1):
                value = -value
            values.append(value)
    al = pd.DataFrame(data={
        'date': dates,
        'description': descriptions,
        'value': values
    })
    al['source'] = 'Alelo'
    return al


def get_previous_entries(doc_name: str):
    gspread_connector = gspread.service_account()
    spreadsheet = gspread_connector.open(doc_name)
    valid_data_sheet_pattern = re.compile(r'20\d\d-[01]\d')
    data_sheet_list = [
        ws.title for ws in spreadsheet.worksheets()
        if valid_data_sheet_pattern.fullmatch(ws.title)
    ]
    total_list_contents = list()
    for data_sheet in data_sheet_list:
        logging.info(f'From {doc_name}, reading sheet {data_sheet}')
        data_sheet_contents = read_gsheet(doc_name, data_sheet)
        total_list_contents.append(data_sheet_contents)
    previous_entries = pd.concat(total_list_contents)
    return previous_entries


def read_gsheet(doc_name: str, sheet_name: str):
    gspread_connector = gspread.service_account()
    spreadsheet = gspread_connector.open(doc_name)
    prev_worksheet = spreadsheet.worksheet(sheet_name)
    prev = pd.DataFrame(
        prev_worksheet.get_all_records(
            value_render_option=['UNFORMATTED_VALUE']))
    prev = prev.rename(
        {
            'Data': '(timestamp)',
            'Categoria': 'category',
            'Nome': 'description',
            'Valor': 'value',
            'Fonte': 'source',
            'Categoria Original': 'original_category'
        },
        axis='columns')
    prev['date'] = pd.to_datetime(prev['(timestamp)'],
                                  unit='d',
                                  origin='1899-12-30')
    for column_name in list(prev.keys()):
        if column_name.startswith('('):
            prev.drop([column_name], axis=1, inplace=True)
    prev['value'] = prev['value'].apply(lambda x: str(x).replace(
        'R$ ', '').replace('.', '').replace(',', '.')).astype('float')
    return prev


def date_filter(entries_unfiltered, start_date: str, end_date: str):
    entries_unfiltered.insert(0, 'order', range(len(entries_unfiltered)))
    entries_unfiltered = entries_unfiltered.sort_values(
        by=['date', 'order']).drop(['order'], axis=1)
    current_entries = entries_unfiltered[
        (entries_unfiltered['date'] >= pd.to_datetime(start_date))
        & (entries_unfiltered['date'] < pd.to_datetime(end_date))]
    current_entries = current_entries.reset_index().drop(['index'], axis=1)
    return current_entries


def preprocess_and_split(prev, current_entries):
    prevf = pd.concat([prev, current_entries]).fillna('')

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

    desired_colums = [
        'wd', 'md', 'month', 'orcat', 'catf', 'descf', 'value', 'sourcef'
    ]
    prevf = prevf.filter(desired_colums)

    Y = prevf['catf'].values
    prevf.drop('catf', axis=1, inplace=True)

    colunas = prevf.columns
    norm = preprocessing.Normalizer()
    dados_normalizados = norm.transform(prevf.values)
    df_norm = pd.DataFrame(dados_normalizados, columns=colunas)
    X = dados_normalizados

    x_treino = X[0:len(prev)]
    y_treino = Y[0:len(prev)]
    x_teste = X[len(prev):]
    y_teste = Y[len(prev):]
    return x_treino, x_teste, y_treino, y_teste, le_cat


def write_gsheet(new_entries, doc_name: str, sheet_name: str) -> None:
    gspread_connector = gspread.service_account()
    spreadsheet = gspread_connector.open(doc_name)
    worksheet_list = [ws.title for ws in spreadsheet.worksheets()]
    if sheet_name in worksheet_list:
        logging.info(f'Updating {sheet_name} in {doc_name}')
        worksheet = spreadsheet.worksheet(sheet_name)
    else:
        logging.info(f'Creating {sheet_name} in {doc_name}')
        worksheet = spreadsheet.add_worksheet(title=sheet_name,
                                              rows=len(new_entries.index) + 1,
                                              cols=7)
    worksheet.update('A1', [[
        'Data', 'Categoria', 'Nome', 'Valor', 'Fonte', 'Categoria Original',
        'Comentário'
    ]])
    new_entries['date'] = new_entries['date'].astype(str)
    worksheet.update('A2',
                     new_entries.values.tolist(),
                     value_input_option='USER_ENTERED')


def classify(method, x_treino, x_teste, y_treino, le_cat):
    method.fit(x_treino, y_treino)
    method.score(x_treino, y_treino)
    pred_knn = method.predict(x_teste)
    return le_cat.inverse_transform(pred_knn)


if __name__ == '__main__':
    main()
