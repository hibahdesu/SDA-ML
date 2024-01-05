#Importing Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

#Reading the dataset
df = pd.read_csv('train.csv')

#Seeing the head of the dataset
print(df.head())

print(df.shape)

print(df.describe())

print(df.info())

print(df.columns)

p = sns.pairplot(df[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')
# print(p)
plt.show()


tensor_df = tf.constant(df)
tensor_df = tf.cast(tensor_df, tf.float32)

print(tensor_df)
