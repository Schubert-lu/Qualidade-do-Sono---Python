#%% Instalando Pacotes

!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp


#%% Importanto Pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica|
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
from playsound import playsound # reprodução de sons
import pingouin as pg # outro modo para obtenção de matrizes de correlações
import emojis # inserção de emojis em gráficos
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
import plotly.express as px
from matplotlib import font_manager

#%% Importando Banco de Dados

df = pd.read_excel("Qualidade de Sono.xlsx")
## Importado de: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

#%% Mudando fonte para Arial

plt.rcParams['font.family'] = 'Arial'

#%% Organinzando colunas

df = df.rename(columns={'Person ID': 'Pessoa_ID',
                        'Gender': 'Sexo',
                        'Age': 'Idade',
                        'Occupation': 'Cargo',
                        'Sleep Duration': 'Duracao_Sono_h',
                        'Quality of Sleep': 'Qualidade_Sono',
                        'Physical Activity Level': 'Nível_Atividade_Fisica_h',
                        'Stress Level': 'Nivel_Estresse',
                        'BMI Category': 'Categoria_IMC',
                        'Blood Pressure': 'Pressao_Arterial',
                        'Heart Rate': 'Frequencia_Cardiaca',
                        'Daily Steps': 'Passos_Diarios',
                        'Sleep Disorder': 'Disturbio_do_Sono'})


#%% Informacoes da Base de Dados

print(df.info())


#%%Estatisticas descritivas das variaveis

tab_desc = df.describe()

#%% Separando as profissoes por área

area = {
    'Nurse': 'Saúde',
    'Doctor': 'Saúde',
    'Engineer': 'Tecnologia',
    'Software Engineer': 'Tecnologia',
    'Scientist': 'Tecnologia',
    'Lawyer': 'Direito',
    'Teacher': 'Educação',
    'Salesperson': 'Comercial',
    'Sales Representative': 'Comercial',
    'Accountant': 'Comercial',
    'Manager': 'Comercial'
}

df['Area'] = df['Cargo'].map(area)

#%% Criando faixas etárias

limites_faixas_etarias = [26, 35, 45, 55, 60]
faixa_etaria = ['27-35', '36-45', '46-55', '55+']

df['faixa_etaria'] = pd.cut(df['Idade'], bins = limites_faixas_etarias, labels= faixa_etaria)

#%% Convertendo Nivel de Atividade Fisica pra horas (esta em minutos)

df['Nível_Atividade_Fisica_h'] = df['Nível_Atividade_Fisica_h'] /60

df['Nível_Atividade_Fisica_h'] = df['Nível_Atividade_Fisica_h'].round(1)

#%% Preenchendo colunas vazias

df['Disturbio_do_Sono'].fillna('No Sleep Disorder',inplace=True)

#%% Análises demograficas

#%% Sexo

cont_sexo = df['Sexo'].value_counts()

percent_sexo = (cont_sexo / cont_sexo.sum()) * 100

percent_sexo = percent_sexo.round(2)

resumo_sexo = pd.DataFrame({'Contagem' : cont_sexo, '%' : percent_sexo})

print(resumo_sexo)

#%% Peso

cont_peso = df['Categoria_IMC'].value_counts()

percent_peso = (cont_peso / cont_peso.sum()) * 100

percent_peso = percent_peso.round(1)

resumo_peso = pd.DataFrame({'Contagem' : cont_peso, '%' : percent_peso})

print(resumo_peso)

#%% Pessoas com disturbio de sono

cont_disturbio = df['Disturbio_do_Sono'].value_counts()

percent_disturbio = (cont_disturbio / cont_disturbio.sum ()) * 100

percent_disturbio = percent_disturbio.round(1)

resumo_peso = pd.DataFrame({'Contagem' : cont_disturbio, '%' : percent_disturbio})

print(resumo_peso)

#%% Profissoes analisadas

cont_profissoes = df['Cargo'].value_counts()

print(cont_profissoes)

#%%  Analise por área

cont_area = df['Area'].value_counts()

percent_area = (cont_area / cont_area.sum ()) * 100

percent_area = percent_area.round(1)

resumo_area = pd.DataFrame({'Contagem' : cont_area, '%' : percent_area})

print(resumo_area)

colors = sns.color_palette('pastel')[0:5]

plt.figure(figsize = (10, 9))
wedges, texts, autotexts = plt.pie(cont_area, colors = colors,autopct='%1.1f%%', startangle=140)

# Equaliza o aspecto do gráfico para que o círculo seja desenhado como um círculo
plt.axis('equal')  

# Adiciona a legenda
plt.legend(wedges, cont_area.index, title="Áreas", loc="upper right")

plt.title('Distribuição de Áreas', fontsize=16, color='Black', loc='center', pad=30)
plt.show()

#%% Qualidade do sono por área (Descontando problemas de Sono)

media_sono_area = df.groupby('Area')['Qualidade_Sono'].mean().reset_index()

print(media_sono_area)

plt.figure(figsize = (10, 9))
bars = plt.bar(media_sono_area['Area'], media_sono_area['Qualidade_Sono'], color=colors)

plt.title('Média da Qualidade do Sono por Área', fontsize=16, color='Black', loc='center')
plt.xlabel('Área', fontsize=8)
plt.ylabel('Média da Qualidade do Sono', fontsize=8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

#%% Qualidade do sono relacionado a disturbios do sono

media_sono_semdisturb = df[df['Disturbio_do_Sono'].isnull()]['Qualidade_Sono'].mean()

print('Media sem disturbio: ',  media_sono_semdisturb)

media_sono_comdisturb = df.groupby('Disturbio_do_Sono')['Qualidade_Sono'].mean().reset_index()

print('Media com disturbio: \n',  media_sono_comdisturb)

#%%

plt.figure(figsize=(12, 8))

sns.scatterplot(data=df, x='Cargo', y='Duracao_Sono_h', hue='Disturbio_do_Sono', palette='Set1')

plt.title('Duração do Sono por Cargo e Distúrbio do Sono', fontsize=18, pad=20,)
plt.xlabel('Cargo', fontsize=12)
plt.ylabel('Duração do Sono (h)', fontsize=12)

plt.grid(True)

plt.xticks(rotation=45)
plt.show()

#%%

plt.figure(figsize=(12, 8))

sns.scatterplot(data=df, x='Categoria_IMC', y='Frequencia_Cardiaca', hue='Disturbio_do_Sono', palette='Set1')

plt.title('Duração do Sono por Cargo e Distúrbio do Sono', fontsize=18, pad=20,)
plt.xlabel('Categoria IMC', fontsize=12)
plt.ylabel('Frequencia Cardiaca', fontsize=12)

plt.grid(True)

plt.xticks()
plt.show()

#%% 
plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='Sexo', hue='Disturbio_do_Sono', weights='Qualidade_Sono', 
             multiple='stack', palette='Set1',shrink=0.9,edgecolor=None)

plt.title('Qualidade do Sono por Gênero e Distúrbio do Sono', fontsize=16)
plt.xlabel('Gênero', fontsize=14)
plt.ylabel('Qualidade do Sono', fontsize=14)

plt.show()

#%% 

df['Pressao_Arterial'] = pd.Categorical(df['Pressao_Arterial'], ordered=True)
plt.figure(figsize=(12, 10))

sns.histplot(data=df, x='Pressao_Arterial', hue='Disturbio_do_Sono', weights='Frequencia_Cardiaca', 
             multiple='stack', palette='Set1',edgecolor=None,shrink=0.9)

plt.title('Influencia da Pressao Arterial e Frequencia cardiaca nos Disturios de Sono', fontsize=16)
plt.xlabel('Pressao Arterial', fontsize=14)
plt.ylabel('Frequencia Cardiaca', fontsize=14)

plt.xticks(rotation=270)
plt.show()

#%%

df_corr = df.drop(columns=['Sexo', 'Cargo', 'Categoria_IMC', 'Pressao_Arterial', 'Disturbio_do_Sono', 'Area', 'faixa_etaria']

)

#%% Análise gráfica das correlações de Pearson

matriz_corr2 = df_corr.corr()

sns.heatmap(matriz_corr2, annot=True, 
            cmap = plt.cm.Purples,
            annot_kws={'size':7})