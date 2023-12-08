import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('Анализ склонности клиентов откликнуться на маркетинговое предложение банка')
st.subheader('В рамках данного анализ рассматривается влияние различных характеристик клиентов на вероятность отклика на предложение банка')
st.write('Подготовил: Поспелов Артем Алексеевич')

#Выгрузим уже предобработанные данные
df = st.cache_data(pd.read_csv)("/Users/artempospelov/Desktop/bank_streamlit_hw/df_processed_hw1_streamlit.csv")

#Оставим только необходиме столбцы
col = ['AGREEMENT_RK', 'TARGET', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
       'GENDER', 'AGE', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']
df = df[col]

#Описательные статистики
st.subheader('Описательные статистики:')

st.write(df.iloc[:, 1:].describe())
st.write('В итоговом датасете у нас есть бинарные признаки: GENDER, SOCSTATUS_WORK_FL, SOCSTATUS_PENS_FL; все эти признаки достаточно вариабельны, поэтому нет смысла их удалять')
st.write('При этом в непрерывных признаках (кроме AGE) наблюдаются выбросы (в сторону больших значений) ввиду сверхбольшой разницы между max и q75, стоит рассмотреть их гистограммы распределений')
st.write('Целевая переменная - TARGET (бинарная)')
st.write('В данных не было пропусков, при этом были дубликаты по AGREEMENT_RK, которые были удалены (представлено в ноутбуке обработки данных)')

#Pie-charts бинарных признаков
st.subheader('Pie-charts бинарных признаков:')

bin_cols = {
            'TARGET': ['отклика не было', 'отклик был зарегистрирован'],
            'GENDER': ['Женщина', 'Мужчина'],
            'SOCSTATUS_WORK_FL': ['не работает', 'работает'],
            'SOCSTATUS_PENS_FL': ['не пенсионер', 'пенсионер'],
            }

feat = st.selectbox(
    'Выберите признак:',
    list(bin_cols.keys()))


def func(pct, allvals):
    absolute = int(np.round(pct / 100. * df[allvals].count()))
    return f"{pct:.1f}%\n({absolute:d})"


df_feat = df.groupby(by=feat).AGREEMENT_RK.count().reset_index()
sizes = df_feat.AGREEMENT_RK

fig_pie, ax_pie = plt.subplots(figsize=(3, 3))

ax_pie.pie(sizes, labels=bin_cols.get(feat), autopct=lambda pct: func(pct, feat),
           pctdistance=.5, labeldistance=1.1,
           colors=['darkorange', 'cornflowerblue'])
plt.title(f'Распределение значений по {feat}')
st.pyplot(fig_pie)

st.write('Отклик на маркетинговое предложение составляет 12% от общих показов')
st.write('Среди тех, кому показали предложение 65% составили мужчины')
st.write('Более 90% тех, кому показали предложение работают')
st.write('Более 86,5% тех, кому показали предложение НЕ пенсионеры')
st.subheader('Гистограммы распределения признаков:')

feat_h = st.selectbox(
    'Выберите признак:',
    df.columns[5:].to_list())

fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df[feat_h], bins=20)
plt.title(f'Распределение значений по {feat_h}')
st.pyplot(fig_hist)

st.subheader('Выводы по каждому признаку:')
st.write('AGE: предложение чаще показывалось молодым, чем пожилым клиентам')
st.write('CHILD_TOTAL: в основном у клиентов не более 2-3 детей, также встречаются клиенты с > 4 детьми, однако это редкие наблюдения ')
st.write('PERSONAL_INCOME: в основном заработок клиентов не превышает 50_000, хотя концентрируется около 12_000 - 15_000')
st.write('LOAN_NUM_TOTAL и LOAN_NUM_CLOSED: резко убывает при этом редко превышает 4, концентрируется около 0-1')




st.subheader('Связи переменных с целевой (Target):')

col_con = ['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER', 'AGE',
           'CHILD_TOTAL', 'DEPENDANTS', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']

feat_con = st.selectbox('Выберите признак:', col_con)
# Группируем данные по столбцу gender и вычисляем среднее значение target в каждой группе
grouped_df = df.groupby(feat_con)['TARGET'].mean()

fig_con, ax_con = plt.subplots()
ax_con.bar(grouped_df.index, grouped_df.values)
plt.xlabel(f'{feat_con}')
plt.ylabel('Mean Target')
plt.title(f'Распределение значений по {feat_con}')
st.pyplot(fig_con)

st.subheader('Выводы по каждому признаку:')
st.write('GENDER: Женщины откликаются несколько чаще на предложение, чем мужчины')
st.write('AGE: Молодые клиенты откликаются чаще на предложение, чем пожилые ')
st.write('CHILD_TOTAL и DEPENDANTS: кол-во детей не оказывает влияние на частоту откликов, так как изменение доли откликов наблюдается только в экстремальных наблюдениях')
st.write('SOCSTATUS_WORK_FL: работающие люди значительно чаще откликаются на маркетиновгове предложение')
st.write('SOCSTATUS_PENS_FL: пенсионеры значительно реже откликаются на маркетиновгове предложение')
st.write('LOAN_NUM_TOTAL и LOAN_NUM_CLOSED: чем меньше кредитов было у клиента тем он более склонен откликнуться на предложение (на экстремальные значения не стоит обращать внимание)')

st.subheader('Связь переменной PERSONAL_INCOME с целевой (Target):')

# Разбиваем переменную income на 5 бакетов и добавляем новый столбец в DataFrame

n_buck = st.slider('Выберите признак:', 1, 10, 5)

df['income_bucket'] = pd.qcut(df['PERSONAL_INCOME'], n_buck)

# Группируем данные по столбцу income_bucket и вычисляем среднее значение target в каждой группе
grouped_df_1 = df.groupby('income_bucket')['TARGET'].mean()

fig_con_cont, ax_con_cont = plt.subplots(figsize=(n_buck * 2, n_buck))
ax_con_cont.bar(grouped_df_1.index.astype(str), grouped_df_1.values)
plt.xticks(fontsize=4 + n_buck, rotation=45)
plt.yticks(fontsize=6 + n_buck)
plt.xlabel('Income Bucket', fontsize=7 + n_buck)
plt.ylabel('Mean Target', fontsize=7 + n_buck)
plt.title(f'Среднее значение target по PERSONAL_INCOME', fontsize=10 + n_buck)
st.pyplot(fig_con_cont)

st.write('При небольшом кол-ве разбиений заметно, что повышение дохода положительно влияет на долю откликов, однако при большем числе разбиений связь ослабевает')

st.subheader('Корреляционная матрица исходных признаков:')

col_corr = ['TARGET', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER', 'AGE',
            'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']
df_fil = df[col_corr].copy()

q_dict = {}
for col in df_fil.columns:
    q_dict[col] = df[col].quantile(0.95)

for col, q in q_dict.items():
    df_fil = df_fil[df_fil[col] <= q]

fig_corr, ax_corr = plt.subplots()
sns.heatmap(df_fil.corr(), cmap="coolwarm", annot=True, fmt=".1f", ax=ax_corr)
st.write(fig_corr)

st.write('Нет признаков, которые значительно коррелируют с TARGET')
st.write('Сильно коррелируют между собой LOAN_NUM_TOTAL и LOAN_NUM_CLOSED')