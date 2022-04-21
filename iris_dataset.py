#importing required packages
from urllib.request import urlopen

import numpy as np

from sys import argv

import math
import statistics

# importing pandas
import pandas as pd

#specifiy url to download data file from   
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#defining download function
def download(iris_url):
    response = urlopen(iris_url)
    data = response.read()      #reads data from url
    txt_str = str(data)         #converts the entire data to a string
    datanew = txt_str[2:-3]     

    #splitting the string to lines
    lines = datanew.split("\\n")

    #specify destination folder to store downloaded file
    dest_url = "/mnt/c/users/victor/desktop/projects/Data_Analytics/iris_data.txt"
    
    fx = open(dest_url,"w")
    for line in lines:
        fx.write(line + "\n")
    fx.close()

download(data_url)

#loading entire Iris data set into a single data frame
df = pd.read_csv("iris_data.txt", sep=",", header=None, 
        names=['sepal length', 'sepal width', 'petal length', 'petal width', 'label'])

#Converting data frame to array(Numpy) of individual attributes
data_sepal_len_arr = df['sepal length'].to_numpy()
data_sepal_wid_arr = df['sepal width'].to_numpy()
data_petal_len_arr = df['petal length'].to_numpy()
data_petal_wid_arr = df['petal width'].to_numpy()


#Iris-setosa data frame for individual attributes
setosa_df = df.loc[df['label'] == 'Iris-setosa']

setosa_sepal_len_arr = setosa_df['sepal length'].to_numpy()
setosa_sepal_wid_arr = setosa_df['sepal width'].to_numpy()
setosa_petal_len_arr = setosa_df['petal length'].to_numpy()
setosa_petal_wid_arr = setosa_df['petal width'].to_numpy()


#Iris-versicolor data frame
versicolor_df = df.loc[df['label'] == 'Iris-versicolor']

versicolor_sepal_len_arr = versicolor_df['sepal length'].to_numpy()
versicolor_sepal_wid_arr = versicolor_df['sepal width'].to_numpy()
versicolor_petal_len_arr = versicolor_df['petal length'].to_numpy()
versicolor_petal_wid_arr = versicolor_df['petal width'].to_numpy()

#Iris-virginica data frame
virginica_df = df.loc[df['label'] == 'Iris-virginica']

virginica_sepal_len_arr = virginica_df['sepal length'].to_numpy()
virginica_sepal_wid_arr = virginica_df['sepal width'].to_numpy()
virginica_petal_len_arr = virginica_df['petal length'].to_numpy()
virginica_petal_wid_arr = virginica_df['petal width'].to_numpy()


#Getting Mean Std Dev. Max and Min for the Iris Dataset with the three classes

def iris_data():
    import math 
    import statistics #for standard deviation calculations

    print("\nCOMPUTATIONS FOR DATASET WITH ALL THREE CLASSES IN THE DATASHEET\n")
    n_data = len(data_sepal_len_arr)
    
    #Computing Sepal Length Mean, S.D, Max and Min
    mean_data_sepal_len = sum(data_sepal_len_arr) / n_data
    std_dev_data_sepal_len = statistics.pstdev(data_sepal_len_arr)
    data_max_sepal_len = np.max(data_sepal_len_arr)
    data_min_sepal_len = np.min(data_sepal_len_arr)
    print("For the Iris Dataset Sepal Length.\nMean: ","%.2f" % mean_data_sepal_len, 
    "\nStandard Deviation: ","%.2f" % std_dev_data_sepal_len,
    "\nMaximum: ", data_max_sepal_len,
    "\nMinimum: ", data_min_sepal_len)
    print("\n")

    #Computing Sepal Width Mean, S.D, Max and Min
    mean_data_sepal_wid = sum(data_sepal_wid_arr) / n_data
    std_dev_data_sepal_wid = statistics.pstdev(data_sepal_wid_arr)
    max_data_sepal_wid = np.max(data_sepal_wid_arr)
    min_data_sepal_wid = np.min(data_sepal_wid_arr)
    print("For the Iris Dataset Sepal Width.\nMean: ","%.2f" % mean_data_sepal_wid, 
    "\nStandard Deviation: ","%.2f" % std_dev_data_sepal_wid,
    "\nMaximum: ", max_data_sepal_wid,
    "\nMinimum: ", min_data_sepal_wid)
    print("\n")

    #Computing Petal Length Mean, S.D, Max and Min
    mean_data_petal_len = sum(data_petal_len_arr) / n_data
    std_dev_data_petal_len = statistics.pstdev(data_petal_len_arr)
    data_max_petal_len = np.max(data_petal_len_arr)
    data_min_petal_len = np.min(data_petal_len_arr)
    print("For the Iris Dataset Petal Length.\nMean: ","%.2f" % mean_data_petal_len, 
    "\nStandard Deviation: ","%.2f" % std_dev_data_petal_len,
    "\nMaximum: ", data_max_petal_len,
    "\nMinimum: ", data_min_petal_len)
    print("\n")

    #Computing Petal Width Mean, S.D, Max and Min
    mean_data_petal_wid = sum(data_petal_wid_arr) / n_data
    std_dev_data_petal_wid = statistics.pstdev(data_petal_wid_arr)
    max_data_petal_wid = np.max(data_petal_wid_arr)
    min_data_petal_wid = np.min(data_petal_wid_arr)
    print("For the Iris Dataset Petal Width.\nMean: ","%.2f" % mean_data_petal_wid, 
    "\nStandard Deviation: ","%.2f" % std_dev_data_petal_wid,
    "\nMaximum: ", max_data_petal_wid,
    "\nMinimum: ", min_data_petal_wid)
    print("\n")

iris_data()


#For individual classes starting with Iris-Setosa
def setosa_class():
    import math 
    import statistics #for standard deviation calculations

    n_setosa = len(setosa_sepal_len_arr)
    mean_sepal_len = sum(setosa_sepal_len_arr) / n_setosa
    print("\nCLASS IRIS-SETOSA\n")
    print("Mean of Class Setosa sepal length is: ", "%.2f" % mean_sepal_len)
    std_dev_sepal_len = statistics.pstdev(setosa_sepal_len_arr)
    print("Standard Deviation of Class Setosa sepal length is: ", "%.2f" % std_dev_sepal_len)
    max_sepal_len = np.max(setosa_sepal_len_arr)
    print("The Maximum Value of Class Setosa Sepal lenth is: ", max_sepal_len)
    min_sepal_len = np.min(setosa_sepal_len_arr)
    print("The Minimum Value of Class Setosa Sepal lenth is: ", min_sepal_len)
    print("\n")

    mean_sepal_wid = sum(setosa_sepal_wid_arr) / n_setosa
    print("Mean of Class Setosa sepal width is: ", "%.2f" % mean_sepal_wid)
    std_dev_sepal_wid = statistics.pstdev(setosa_sepal_wid_arr)
    print("Standard Deviation of Class Setosa sepal width is: ", "%.2f" % std_dev_sepal_wid)
    max_sepal_wid = np.max(setosa_sepal_wid_arr)
    print("The Maximum Value of Class Setosa Sepal width is: ", max_sepal_wid)
    min_sepal_wid = np.min(setosa_sepal_wid_arr)
    print("The Minimum Value of Class Setosa Sepal wid is: ", min_sepal_wid)
    print("\n")

    mean_petal_len = sum(setosa_petal_len_arr) / n_setosa
    print("Mean of Class Setosa petal length is: ", "%.2f" % mean_petal_len)
    std_dev_petal_len = statistics.pstdev(setosa_petal_len_arr)
    print("Standard Deviation of Class Setosa petal length is: ", "%.2f" % std_dev_petal_len)
    max_petal_len = np.max(setosa_petal_len_arr)
    print("The Maximum Value of Class Setosa petal lenth is: ", max_petal_len)
    min_petal_len = np.min(setosa_petal_len_arr)
    print("The Minimum Value of Class Setosa petal lenth is: ", min_petal_len)
    print("\n")

    mean_petal_wid = sum(setosa_petal_wid_arr) / n_setosa
    print("Mean of Class Setosa petal width is: ", "%.2f" % mean_petal_wid)
    std_dev_petal_wid = statistics.pstdev(setosa_petal_wid_arr)
    print("Standard Deviation of Class Setosa petal width is: ", "%.2f" % std_dev_petal_wid)
    max_petal_wid = np.max(setosa_petal_wid_arr)
    print("The Maximum Value of Class Setosa petal lenth is: ", max_petal_wid)
    min_petal_wid = np.min(setosa_petal_wid_arr)
    print("The Minimum Value of Class Setosa petal lenth is: ", min_petal_wid)
    print("\n")

setosa_class()

#considering class Iris-versicolor only
def versicolor_class():
    import math 
    import statistics 
    print("\nCLASS IRIS-VERSICOLOR\n")

    n_versicolor = len(versicolor_sepal_len_arr)
    mean_sepal_len_versi = sum(versicolor_sepal_len_arr) / n_versicolor
    print("Mean of Class Versicolor sepal length is: ", "%.2f" % mean_sepal_len_versi)
    std_dev_sepal_len_versi = statistics.pstdev(versicolor_sepal_len_arr)
    print("Standard Deviation of Class versicolor sepal length is: ", "%.2f" % std_dev_sepal_len_versi)
    max_sepal_len_versi = np.max(versicolor_sepal_len_arr)
    print("The Maximum Value of Class versicolor Sepal lenth is: ", max_sepal_len_versi)
    min_sepal_len_versi = np.min(versicolor_sepal_len_arr)
    print("The Minimum Value of Class versicolor Sepal lenth is: ", min_sepal_len_versi)
    print("\n")

    mean_sepal_wid_versi = sum(versicolor_sepal_wid_arr) / n_versicolor
    print("Mean of Class versicolor sepal width is: ", "%.2f" % mean_sepal_wid_versi)
    std_dev_sepal_wid_versi = statistics.pstdev(versicolor_sepal_wid_arr)
    print("Standard Deviation of Class versicolor sepal width is: ", "%.2f" % std_dev_sepal_wid_versi)
    max_sepal_wid_versi = np.max(versicolor_sepal_wid_arr)
    print("The Maximum Value of Class versicolor Sepal width is: ", max_sepal_wid_versi)
    min_sepal_wid_versi = np.min(versicolor_sepal_wid_arr)
    print("The Minimum Value of Class versicolor Sepal width is: ", min_sepal_wid_versi)
    print("\n")

    mean_petal_len_versi = sum(versicolor_petal_len_arr) / n_versicolor
    print("Mean of Class versicolor petal length is: ", "%.2f" % mean_petal_len_versi)
    std_dev_petal_len_versi = statistics.pstdev(versicolor_petal_len_arr)
    print("Standard Deviation of Class versicolor petal length is: ", "%.2f" % std_dev_petal_len_versi)
    max_petal_len_versi = np.max(versicolor_petal_len_arr)
    print("The Maximum Value of Class versicolor petal lenth is: ", max_petal_len_versi)
    min_petal_len_versi = np.min(versicolor_petal_len_arr)
    print("The Minimum Value of Class versicolor petal lenth is: ", min_petal_len_versi)
    print("\n")

    mean_petal_wid_versi = sum(versicolor_petal_wid_arr) / n_versicolor
    print("Mean of Class versicolor petal width is: ", "%.2f" % mean_petal_wid_versi)
    std_dev_petal_wid_versi = statistics.pstdev(versicolor_petal_wid_arr)
    print("Standard Deviation of Class versicolor petal width is: ", "%.2f" % std_dev_petal_wid_versi)
    max_petal_wid_versi = np.max(versicolor_petal_wid_arr)
    print("The Maximum Value of Class versicolor petal lenth is: ", max_petal_wid_versi)
    min_petal_wid_versi = np.min(versicolor_petal_wid_arr)
    print("The Minimum Value of Class versicolor petal lenth is: ", min_petal_wid_versi)
    print("\n")

versicolor_class()


#considering only class Iris-virginica
def virginica_class():
    import math 
    import statistics
    print("\nCLASS IRIS-VIRGINICA\n")


    n_virginica = len(virginica_sepal_len_arr)
    mean_sepal_len_virgi = sum(virginica_sepal_len_arr) / n_virginica
    print("Mean of Class virginica sepal length is: ", "%.2f" % mean_sepal_len_virgi)
    std_dev_sepal_len_virgi = statistics.pstdev(virginica_sepal_len_arr)
    print("Standard Deviation of Class virginica sepal length is: ", "%.2f" % std_dev_sepal_len_virgi)
    max_sepal_len_virgi = np.max(virginica_sepal_len_arr)
    print("The Maximum Value of Class virginica Sepal lenth is: ", max_sepal_len_virgi)
    min_sepal_len_virgi = np.min(virginica_sepal_len_arr)
    print("The Minimum Value of Class virginica Sepal lenth is: ", min_sepal_len_virgi)
    print("\n")

    mean_sepal_wid_virgi = sum(virginica_sepal_wid_arr) / n_virginica
    print("Mean of Class virginica sepal width is: ", "%.2f" % mean_sepal_wid_virgi)
    std_dev_sepal_wid_virgi = statistics.pstdev(virginica_sepal_wid_arr)
    print("Standard Deviation of Class virginica sepal width is: ", "%.2f" % std_dev_sepal_wid_virgi)
    max_sepal_wid_virgi = np.max(virginica_sepal_wid_arr)
    print("The Maximum Value of Class virginica Sepal width is: ", max_sepal_wid_virgi)
    min_sepal_wid_virgi = np.min(virginica_sepal_wid_arr)
    print("The Minimum Value of Class virginica Sepal width is: ", min_sepal_wid_virgi)
    print("\n")

    mean_petal_len_virgi = sum(virginica_petal_len_arr) / n_virginica
    print("Mean of Class virginica petal length is: ", "%.2f" % mean_petal_len_virgi)
    std_dev_petal_len_virgi = statistics.pstdev(virginica_petal_len_arr)
    print("Standard Deviation of Class virginica petal length is: ", "%.2f" % std_dev_petal_len_virgi)
    max_petal_len_virgi = np.max(virginica_petal_len_arr)
    print("The Maximum Value of Class virginica petal lenth is: ", max_petal_len_virgi)
    min_petal_len_virgi = np.min(virginica_petal_len_arr)
    print("The Minimum Value of Class virginica petal lenth is: ", min_petal_len_virgi)
    print("\n")

    mean_petal_wid_virgi = sum(virginica_petal_wid_arr) / n_virginica
    print("Mean of Class virginica petal width is: ", "%.2f" % mean_petal_wid_virgi)
    std_dev_petal_wid_virgi = statistics.pstdev(virginica_petal_wid_arr)
    print("Standard Deviation of Class virginica petal width is: ", "%.2f" % std_dev_petal_wid_virgi)
    max_petal_wid_virgi = np.max(virginica_petal_wid_arr)
    print("The Maximum Value of Class virginica petal lenth is: ", max_petal_wid_virgi)
    min_petal_wid_virgi = np.min(virginica_petal_wid_arr)
    print("The Minimum Value of Class virginica petal lenth is: ", min_petal_wid_virgi)
    print("\n")

virginica_class()

