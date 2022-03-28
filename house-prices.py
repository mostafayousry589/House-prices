import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble


data = pd.read_csv("houses-data.csv")
#print(data.head()) 
#print(data.describe())


#number of Bedroom
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')


#Longitude and Latitude of houses
plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


"""
relations between some features and price
"""

#relation between Price and sqft_living
plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")
plt.show()


#relation between price and Longitude
plt.scatter(data.price,data.long)
plt.title("Price vs Longitude")
plt.show()


#relation between price and Latitude
plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")
plt.show()


#relation between price and Bedroom
plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()


#relation between price and Waterfront
plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price")
plt.show()


#relation between price and condition
plt.scatter(data.condition,data.price)
plt.title("condition vs Price")
plt.show()


#relation between price and zipcode
plt.scatter(data.zipcode,data.price)
plt.title("zipcode vs Price")
plt.show()





reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]#Toggle the date data with zero or one  
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)



x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.1,random_state =2)
#reg.fit(x_train,y_train)
#e=reg.score(x_test,y_test)
#print(e)



clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
sc=clf.score(x_test,y_test)
print(sc)




