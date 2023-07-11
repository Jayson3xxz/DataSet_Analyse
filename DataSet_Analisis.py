import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, shapiro, spearmanr
DataSet = pd.read_csv(r"C:\Users\jayson3xxx\Desktop\Training_DataSeTs\data.csv")
DataSet.drop_duplicates(inplace = True)
DataSet.dropna(inplace = True)
print(DataSet.info())
# print(len(DataSet_price.unique()))
#количество уникальных значений цены 1741
#функция распределения предсказываемой переменной
list_price = DataSet['price'].value_counts(normalize = True , ascending= True).copy()
print(list_price)
print(np.sum(list_price))
print(shapiro(list_price) , '\n')
print(kstest(list_price , cdf = 'norm' , N = 500))
#p-value < 0.05 оба теста показали, значит price имеет нормальное распеределение
print(spearmanr(DataSet['price'].copy(),DataSet['bedrooms'].copy()))
#достаточно высокая корреляцимя между стоимостью и числом спален
#удаляем сильно коррелируемые переменные(штат ,город ,страна)
DataSet.drop(labels = ['date','city','statezip','statezip' , 'country'] , axis = 1, inplace = True)
DataSet['street'] = DataSet['street'].factorize()[0]
DataSet['bathrooms'] = DataSet['bathrooms'].factorize()[0]
corr = []
#проверка всех распределений переменных на нормальность
for col in DataSet.columns:
    corr.append(DataSet[col].value_counts(normalize = True,ascending= True).copy())
for i in range(0, len(corr) - 1):
    print((DataSet.columns[i],kstest(corr[i], cdf='norm', N=500)))
#отчистка модели от выбросов
#необходимо использовать метод StandatdScaler()
DST = DataSet[['price' , 'sqft_living','sqft_lot' , 'sqft_above','sqft_basement',
               'yr_built' , 'yr_renovated']].copy().to_numpy()
scaler = StandardScaler().fit(DST)
DST = scaler.transform(DST)
D_S_T = pd.DataFrame(DST , columns = ['price' , 'sqft_living','sqft_lot' , 'sqft_above','sqft_basement',
               'yr_built' , 'yr_renovated'])
DataSet['price'] = D_S_T['price'].copy()
DataSet = DataSet[(-2*DataSet['price'].std() < DataSet['price']) &
                  (DataSet['price'] <2*DataSet['price'].std())]
D_S_T['price'] = DataSet['price'].copy()
D_S = scaler.inverse_transform(D_S_T)
D_S = pd.DataFrame(D_S,columns = ['price' , 'sqft_living','sqft_lot' , 'sqft_above','sqft_basement',
               'yr_built' , 'yr_renovated'])
DataSet['price'] = D_S['price'].copy()
#выбросы были отсеяны осталось около 95 процентов информации
fig = plt.gcf()
fig.set_size_inches(12,12)
#удаляем переменные из расмотрения(из-за низкой корреляции между собой):
#'street','sqft_lot','yr_renovated','yr_built','condition',
#'waterfront'
DataSet.drop(labels = ['street','sqft_lot','yr_renovated','yr_built','condition',
                       'waterfront','sqft_basement'] , axis =1 , inplace= True)
#основные переменные для регрессии имеющие наибольшую корреляцию :
#sqft_living ,sqft_above,bedrooms Нужно проверить на корреляции между собой и проверить на случайность
print(spearmanr(DataSet['sqft_living'],DataSet['sqft_above']),
      spearmanr(DataSet['sqft_living'],DataSet['bedrooms']),
      spearmanr(DataSet['sqft_above'],DataSet['bedrooms']))
#используя корреляцию Спирмена получаем результат:sqft_living корр со всеми признаками
#которые имею высокую корреляцию, эти признаки удаляем
sns.heatmap(DataSet.corr(method = 'spearman'),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
print(spearmanr(DataSet['price'],DataSet['bathrooms']),
      spearmanr(DataSet['price'],DataSet['floors']),
      spearmanr(DataSet['price'],DataSet['view']),
      spearmanr(DataSet['sqft_living'],DataSet['bathrooms']))
DataSet.drop(labels = ['bathrooms'] , axis = 1 , inplace=True)
plt.show()
DataSet.info()
print(DataSet.describe())
#построение функций распределения цены
figure = plt.figure
#Функция распределения имеет смещение вправо(мода и мо смещены)Б но не критично
plt.hist(DataSet['price'] , bins = 30)
plt.show()
#удалим переменную bathrooms, тк она имеет низкую корреляцию
#коробчатые диаграммы и pairplot#
fig2 = plt.figure(figsize = (5,5))
#scatterplot для попрных зависимостей
sns.pairplot(DataSet)
plt.show()
labels = ['price','bedrooms' , 'sqft_living' , 'floors' , 'view' , 'sqft_above']
fig = plt.figure(figsize=(7,7))
for labs in labels:
    plt.boxplot(DataSet[labs])
    plt.xlabel(labs)
    plt.show()