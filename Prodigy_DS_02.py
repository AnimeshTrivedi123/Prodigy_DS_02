{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "ht350gQG1GoT",
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from IPython.display import Image, display\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "Xzk_r7XR1Nhn",
    "tags": []
   },
   "outputs": [],
   "source": [
    "#load the train dataset\n",
    "train = pd.read_csv(\"C:\\\\Users\\\\LENOVO\\\\Downloads\\\\train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 241
    },
    "id": "xIaAiYKM1NmC",
    "outputId": "c2da0102-da99-4c95-e8ce-f7067b60062c",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#inspect the first few rows of the train dataset\n",
    "display(train.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "kwGB7LNH1NpD",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# set the index to passengerId\n",
    "train = train.set_index('PassengerId')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "xbC2lzFq1Nr2",
    "tags": []
   },
   "outputs": [],
   "source": [
    "#load the test dataset\n",
    "test = pd.read_csv(\"C:\\\\Users\\\\LENOVO\\\\Downloads\\\\test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "fmJ4p8nz1Nu1",
    "outputId": "9b51a39a-8b81-4270-a2e9-bd15084a044d",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#inspect the first few rows of the test dataset\n",
    "display(test.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gKENriFA1Nxu",
    "outputId": "24f681a1-bcda-4d1a-c350-69e6d27d6e4b",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 11)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#by calling the shape attribute of the train dataset we can observe that there are 891 observations and 11 columns\n",
    "#in the data set\n",
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 272
    },
    "id": "C56-aONN1N0h",
    "outputId": "65ecce3a-90d5-472f-8655-09426742a8ae",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Survived  Pclass  \\\n",
       "PassengerId                     \n",
       "1                   0       3   \n",
       "2                   1       1   \n",
       "3                   1       3   \n",
       "4                   1       1   \n",
       "5                   0       3   \n",
       "\n",
       "                                                          Name     Sex   Age  \\\n",
       "PassengerId                                                                    \n",
       "1                                      Braund, Mr. Owen Harris    male  22.0   \n",
       "2            Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0   \n",
       "3                                       Heikkinen, Miss. Laina  female  26.0   \n",
       "4                 Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0   \n",
       "5                                     Allen, Mr. William Henry    male  35.0   \n",
       "\n",
       "             SibSp  Parch            Ticket     Fare Cabin Embarked  \n",
       "PassengerId                                                          \n",
       "1                1      0         A/5 21171   7.2500   NaN        S  \n",
       "2                1      0          PC 17599  71.2833   C85        C  \n",
       "3                0      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "4                1      0            113803  53.1000  C123        S  \n",
       "5                0      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check out the data summary\n",
    "# Age, Cabin and Embarked has missing data\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "EmB6VuiA1N3K",
    "outputId": "61d01287-66b0-4239-c225-7107b8adec6a",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>int64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>int64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>object</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>object</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>float64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>int64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>int64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>object</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>float64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>object</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>object</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                0\n",
       "Survived    int64\n",
       "Pclass      int64\n",
       "Name       object\n",
       "Sex        object\n",
       "Age       float64\n",
       "SibSp       int64\n",
       "Parch       int64\n",
       "Ticket     object\n",
       "Fare      float64\n",
       "Cabin      object\n",
       "Embarked   object"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# identify datatypes of the 11 columns, add the stats to the datadict\n",
    "datadict = pd.DataFrame(train.dtypes)\n",
    "datadict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "mKnw_Wbk1N6E",
    "outputId": "36f35d8d-120c-4300-8dd6-9caf8691ad79",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>MissingVal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>float64</td>\n",
       "      <td>177</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>float64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>object</td>\n",
       "      <td>687</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>object</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                0  MissingVal\n",
       "Survived    int64           0\n",
       "Pclass      int64           0\n",
       "Name       object           0\n",
       "Sex        object           0\n",
       "Age       float64         177\n",
       "SibSp       int64           0\n",
       "Parch       int64           0\n",
       "Ticket     object           0\n",
       "Fare      float64           0\n",
       "Cabin      object         687\n",
       "Embarked   object           2"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# identify missing values of the 11 columns,add the stats to the datadict\n",
    "datadict['MissingVal'] = train.isnull().sum()\n",
    "datadict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "CDU9yR6M1N8k",
    "outputId": "06cc9d64-2c93-432b-d43a-e94e63049715",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>MissingVal</th>\n",
       "      <th>NUnique</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>float64</td>\n",
       "      <td>177</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>681</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>float64</td>\n",
       "      <td>0</td>\n",
       "      <td>248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>object</td>\n",
       "      <td>687</td>\n",
       "      <td>147</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>object</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                0  MissingVal  NUnique\n",
       "Survived    int64           0        2\n",
       "Pclass      int64           0        3\n",
       "Name       object           0      891\n",
       "Sex        object           0        2\n",
       "Age       float64         177       88\n",
       "SibSp       int64           0        7\n",
       "Parch       int64           0        7\n",
       "Ticket     object           0      681\n",
       "Fare      float64           0      248\n",
       "Cabin      object         687      147\n",
       "Embarked   object           2        3"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Identify number of unique values, For object nunique will the number of levels\n",
    "# Add the stats the data dict\n",
    "datadict['NUnique']=train.nunique()\n",
    "datadict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "vNXEIZWS2qlC",
    "outputId": "ea75c3b3-af03-4a3c-ca27-d66f229ed064",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>MissingVal</th>\n",
       "      <th>NUnique</th>\n",
       "      <th>Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>float64</td>\n",
       "      <td>177</td>\n",
       "      <td>88</td>\n",
       "      <td>714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>681</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>float64</td>\n",
       "      <td>0</td>\n",
       "      <td>248</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>object</td>\n",
       "      <td>687</td>\n",
       "      <td>147</td>\n",
       "      <td>204</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>object</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>889</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                0  MissingVal  NUnique  Count\n",
       "Survived    int64           0        2    891\n",
       "Pclass      int64           0        3    891\n",
       "Name       object           0      891    891\n",
       "Sex        object           0        2    891\n",
       "Age       float64         177       88    714\n",
       "SibSp       int64           0        7    891\n",
       "Parch       int64           0        7    891\n",
       "Ticket     object           0      681    891\n",
       "Fare      float64           0      248    891\n",
       "Cabin      object         687      147    204\n",
       "Embarked   object           2        3    889"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Identify the count for each variable, add the stats to datadict\n",
    "datadict['Count']=train.count()\n",
    "datadict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 394
    },
    "id": "zcgZLiwT2qn6",
    "outputId": "7d6d69f9-eeb7-4a92-f9a5-352881500444",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>DataType</th>\n",
       "      <th>MissingVal</th>\n",
       "      <th>NUnique</th>\n",
       "      <th>Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>float64</td>\n",
       "      <td>177</td>\n",
       "      <td>88</td>\n",
       "      <td>714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>int64</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>object</td>\n",
       "      <td>0</td>\n",
       "      <td>681</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>float64</td>\n",
       "      <td>0</td>\n",
       "      <td>248</td>\n",
       "      <td>891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>object</td>\n",
       "      <td>687</td>\n",
       "      <td>147</td>\n",
       "      <td>204</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>object</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>889</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         DataType  MissingVal  NUnique  Count\n",
       "Survived    int64           0        2    891\n",
       "Pclass      int64           0        3    891\n",
       "Name       object           0      891    891\n",
       "Sex        object           0        2    891\n",
       "Age       float64         177       88    714\n",
       "SibSp       int64           0        7    891\n",
       "Parch       int64           0        7    891\n",
       "Ticket     object           0      681    891\n",
       "Fare      float64           0      248    891\n",
       "Cabin      object         687      147    204\n",
       "Embarked   object           2        3    889"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# rename the 0 column\n",
    "datadict = datadict.rename(columns={0:'DataType'})\n",
    "datadict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 175
    },
    "id": "CsrF2mr62qq0",
    "outputId": "836ef63b-91db-4314-a1bf-6bbfcbff13c5",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "      <td>204</td>\n",
       "      <td>889</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>891</td>\n",
       "      <td>2</td>\n",
       "      <td>681</td>\n",
       "      <td>147</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>347082</td>\n",
       "      <td>B96 B98</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>1</td>\n",
       "      <td>577</td>\n",
       "      <td>7</td>\n",
       "      <td>4</td>\n",
       "      <td>644</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           Name   Sex  Ticket    Cabin Embarked\n",
       "count                       891   891     891      204      889\n",
       "unique                      891     2     681      147        3\n",
       "top     Braund, Mr. Owen Harris  male  347082  B96 B98        S\n",
       "freq                          1   577       7        4      644"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# get discripte statistcs on \"object\" datatypes\n",
    "train.describe(include=['object'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 300
    },
    "id": "BouU1RRL2qti",
    "outputId": "646ea2ab-f2cc-482f-8935-3732a57e96ae",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Survived      Pclass         Age       SibSp       Parch        Fare\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208\n",
       "std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429\n",
       "min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000\n",
       "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
       "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
       "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# get discriptive statistcs on \"number\" datatypes\n",
    "train.describe(include=['number'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "WGskBoNP2qwA",
    "outputId": "9dd05ef2-8553-4097-fb7e-efd657c9437e",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "0    0.616162\n",
       "1    0.383838\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.Survived.value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 724
    },
    "id": "hc45dklm2qyt",
    "outputId": "4447d120-ecc1-4564-c65f-789addc8353f",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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L3+pd999t8dzH2k6vfLKK/rWt77lccxqtfblNHCaTubf6ujfj55+h775FeypU6cqJiZGq1evlsPh0KFDhxQfH6+Ojo5ec5k6daoWL17c7diwYcNObmLwW3zOPHX+ev3lj9cstbW1ampqUmJiojnW1dWlt99+WytWrJDb7VZAQIDP8vumsLAwJSQk6KOPPvJpHg888IB++tOf6pZbbpF0uFH62WefqaCggGYkvCcoKEiJiYmqrKzUD37wA3O8srJSN9xwgw8zAwD4imEYmjdvnioqKrRp0ybFxsb6OqVzhmEYcrvdvk4D56g//elPSk5OVlZWljn2ySef9Bp/8cUXy2q1ateuXXwl28/117/Vnj179OGHH+qZZ57R97//fUk67kquyy67TL///e914YUX8oTvsxifM0/emXb95Q/XLKmpqd2eUP0f//Ef+s53vqMHH3zQbxqRkuR2u/Xhhx+atdJXDhw4oAEDPB8jExAQ0Ot9fvsaVR+m+fPn6/bbb9f48eM1YcIErVq1Srt27dLdd9/t69TgQ21tbfr444/N/fr6em3btk1RUVG64IILfJgZzmX8XnrH3LlzVVZWpj/84Q8KDw83VzVERkYqJCTEx9mdPR566CFNnjxZMTEx2rdvn8rLy7Vp0yatX7/e16nhHHXRRRfpueee0+uvv67Y2FitXbtWNTU1vX4gDg8PV25urn7yk5/o0KFDuuqqq9Ta2qrq6moNGjTIKysscGL6699q8ODBio6O1qpVqzRs2DDt2rVLP/3pT4/5mrlz52r16tW69dZb9cADD2jIkCH6+OOPVV5ertWrV/tV8wCnx18/Z/rr9aQ/X3/56zVLeHh4t9tBhIWFKTo62uf32szNzdXUqVN1wQUXqKmpSYsWLVJra6vP/79x6tSpeuyxx3TBBRdozJgxeu+997R06VLdeeedXnl/mpEw3XzzzdqzZ48effRRNTY2Kj4+Xq+++qpGjBjh69TgQ1u3btU111xj7h+5r8WsWbNUUlLio6xwruP30jtWrlwpSZo4caLH+LPPPqs77rjD+wmdpb788kvdfvvtamxsVGRkpMaOHav169d3u6cb4C133323tm3bpptvvlkWi0W33nqrsrKy9Nprr/X6mv/8z//U0KFDVVBQoE8//VTnnXeeLrvsMj300ENezBwnoj/+rQYMGKDy8nLdd999io+P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qFGnmT3ORIG+TgA4UXv27NFNN92kO++8U2PHjlV4eLi2bt2qwsJC3XDDDQoJCdEVV1yhxx9/XBdeeKG++uor/exnPzvh80+fPl333HOP7rnnHl1zzTX61re+dcz4lJQU2Ww2zZw5UxdeeKGuuOIK89jMmTP1y1/+UjfccIMeffRRDR8+XLt27dKLL76oBx54QMOHD9f999+vxx9/XHFxcfrud7+rpUuX8jQxAF5x7733qqioSLfccovy8vIUGRmpLVu26Hvf+55Gjx7tETt48GBFR0dr1apVGjZsmHbt2qWf/vSnHjHLli3TsGHDdOmll2rAgAH67W9/K7vdrvPOO08lJSXq6upSUlKSQkNDtXbtWoWEhGjEiBHenDIA9Op415hpaWmaMGGCbrzxRi1evFijR4/WF198oVdffVU33nijxo8fr4ULF6qlpUVPPvmkBg0apNdee02zZ8/WH//4R19PDwBMx6t3vXnqqafMz63Lli1Tc3Oz7rzzTkmHn5R94MABXXfddRoxYoT27t2rJ598Up2dnUpPTzfPERwcrFmzZum//uu/1Nraqvvuu0+ZmZmy2+39Pm/4H5qROGMMGjRISUlJWrZsmT755BN1dnYqJiZGd911lx566CFJ0q9+9SvdeeedGj9+vEaPHq3CwsITfophRESEpk6dqt/+9rf61a9+ddx4i8WiW2+9Vb/85S/18MMPexwLDQ3V22+/rQcffFDTp0/Xvn379K1vfUupqamKiIiQJOXk5KixsVF33HGHBgwYoDvvvFM/+MEP1NLScpI/GQA4OdHR0XrzzTf1wAMPKCUlRQEBAbr00kt15ZVXdosdMGCAysvLdd999yk+Pl6jR4/Wk08+qYkTJ5oxgwYN0uLFi/XRRx8pICBAl19+uV599VUNGDBA5513nh5//HHNnz9fXV1dSkhI0Lp16xQdHe3FGQNA7453jWmxWPTqq69q4cKFuvPOO/WPf/xDdrtdV199tWw2mzZt2qTly5frrbfeMq/z1q5dq7Fjx2rlypW65557fDxDADjsRD5T9+Txxx/X4sWL9d577+nb3/62/vCHP5grw1NSUvTUU0/p3//93/Xll19q8ODBGjdunDZs2ODxR+6LLrpI06dP13XXXad//vOfuu666/T000/3+5zhnyzGN2+wBwAAAAAAAPQRp9Opl1566aRuo4azG/eM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r11zcBth/gbOBhdfF1S06WuLmEHFSBERERERERGRa5BfUY/LgOjgAKKCA9r41Q16n1mCctfxMgzDaOPXF+lYKkaKiIiIiIiIiFyDE5WN06d7xQa3y+v3CAN/XwvF1U73Jjki3krFSBERERERERGRa1BwphjZI7p9ipH+Pmd36P7qeFm7vIdIR1ExUkRERERERETkKvlFJVFdZ+BrsZAUGdRu7zOoeyQA35yqpNJR327vI9LeVIwUEREREREREblKgb1uAiAxMhB/3/Yrs8SGWkmKDMQwYE++RkeK91IxUkRERERERMSL1NfX8+///u+kpKQQFBRE7969ee6553C5XO42hmGQkZFBUlISQUFBjB49mj179piYuvMK7DkIgOSo9pmifa60pAgA9tkrtJGNeC0VI0VERERERES8yIsvvsjvf/97XnnlFfbt28fChQv5zW9+w5IlS9xtFi5cyKJFi3jllVfYvn07NpuNcePGUVFRYWLyzscwDAK79QOgW1T7TdFucl1cKH4+Fspq6rCX17b7+4m0BxUjRURERERERLzIZ599xj333MNdd91Fr169+H//7/8xfvx4vvjiC6CxQLZ48WLmz5/PlClTSEtLY+XKlVRXV7N69WqT03cu+WVOfEOj8LFAQpi13d8vwM+HPvGhAHxdoMKyeCcVI0VERERERES8yK233soHH3zAgQMHAPjqq6/YsmUL3/3udwHIzc3Fbrczfvx493OsViujRo1i69atF31Nh8NBeXl5s0NatqugCoCYIB/82nG9yHPdcGZX7QMnK2hwaaq2eB8/swOIiIiIiIiISOs9/fTTlJWVccMNN+Dr60tDQwMvvPACDz74IAB2ux2AhISEZs9LSEjg6NGjF33NBQsW8Oyzz7Zv8E6oqRgZH+LbYe+ZHB1MSIAvVc4GjhRVcV1caIe9t0hb0MhIEREREfEICxYs4OabbyYsLIz4+Hjuvfde9u/f36zNY489hsViaXYMHz68WRuHw8GsWbOIjY0lJCSEu+++m+PHj3fkrYiItKu3336bVatWsXr1ar788ktWrlzJf/7nf7Jy5cpm7SwWS7PHhmFccK7JvHnzKCsrcx95eXntlr8zybFXA5AQ3HHFSB+LhevPjI7UVG3xRhoZKSIiIiIeISsri5kzZ3LzzTdTX1/P/PnzGT9+PHv37iUkJMTd7jvf+Q7L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RERERDyOYRjMnj2bW2+9lbS0NADsdjsACQkJzdomJCS4r9ntdgICAoiKirpkm/MtWLCAiIgI95GcnNzWtyMiIl6os07ThnPWjVQxUkygYqSIiIiIeJwnnniCXbt28ac//emCaxaLpdljwzAuOHe+y7WZN28eZWVl7iMvL+/qg4uISKdQ12BQW+cCOt8GNnB23ch8bWIjJlAxUkTEwyxYsACLxUJ6err7nHaPFZGuZNasWbz77rv84x//oHv37u7zNpsN4IIRjoWFhe7RkjabDafTSUlJySXbnM9qtRIeHt7sEBGRrq3yTCEy0M8Hq5+vyWnaXtO6kYUVDpwNhslppKtRMVJExINs376d1157jYEDBzY7r91jRaQrMAyDJ554gr/+9a98+OGHpKSkNLuekpKCzWZj06ZN7nNOp5OsrCxGjBgBwJAhQ/D392/WpqCggJycHHcbERGRllQ5Gwt0YZ1wijY0rhsZeWbdyJNV+r4gHUvFSBERD1FZWcnDDz/M66+/3myts/N3j01LS2PlypVUV1ezevVqAPfusS+99BJjx45l8ODBrFq1it27d7N582azbklE5IrMnDmTVatWsXr1asLCwrDb7djtdmpqGqeQNY0az8zMZO3ateTk5PDYY48RHBzMQw89BEBERATTpk1jzpw5fPDBB+zYsYNHHnmEAQMGuHfXFhERaUmls3FkZGfbvOZc3SMbR0eqGCkdTcVIEREPMXPmTO66664Lvixr91gR6SqWLl1KWVkZo0ePJjEx0X28/fbb7jZPPfUU6enpzJgxg6FDh5Kfn8/GjRsJCwtzt3n55Ze59957mTp1KiNHjiQ4OJh169bh69v5ptmJSNeVn5/PI488QkxMDMHBwdx0001kZ2e7r7dmmR+5tMq6xpGR4YGdc2QkQPeoxnUj7ZUqRkrH6rwlfhERL7JmzRq+/PJLtm/ffsG1y+0ee/ToUXebq9k99tlnn22L+CIibcIwWl6zymKxkJGRQUZGxiXbBAYGsmTJEpYsWdKG6UREPEdJSQkjR47kjjvu4P333yc+Pp5vvvmGyMhId5umZX5WrFhB3759ef755xk3bhz79+9v9gscuTj3yMhOOk0bzu6oXVzrwmINMTmNdCUqRoqImCwvL48nn3ySjRs3EhgYeMl27bF77OzZs92Py8vLSU5OvoLkIiIiImKGF198keTk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kVZr675QOlaeNq9ptTtuaCxG/mN/Ib8y+mGxWMyOJF7qqoqRCxYsICEhgR//+MfNzr/xxhucOnWKp59+uk3CiYh4spdeeomePXuqLxSRLmvp0qUAhIeHX3AtIiKC6dOns2jRojb9Av7RRx8RHx9PZGQko0aN4oUXXiA+vnEdq+zsbOrq6hg/fry7fVJSEmlpaWzduvWSxUiHw4HD4XA/Li/XuBgRaT0z+kJpO2d30tZggpbcmhqLv6+Fo0XVHD5dxXVxoWZHEi91VWtG/uEPf+CGG2644HzTb3xERLqCFStWtElfuHTpUgYOHOjekfuWW27h/fffd183DIOMjAySkpIICgpi9OjR7Nmzp9lraI00ETFDTk7OZa+PHz+e7OzsNnu/iRMn8t///d98+OGHvPTSS2zfvp0777zTXUi02+0EBAQQFRXV7HkJCQnY7fZLvu6CBQuIiIhwH8nJyW2WWUQ6v47uC6VtHdfmNa0WavVjeO8YAD7cp1215epdVTHSbreTmJh4wfm4uDgKCgquOZSIiDc4efJkm/SF3bt359e//jVffPEFX3zxBXfeeSf33HOPu+C4cOFCFi1axCuvvML27dux2WyMGzeOiooK92ukp6ezdu1a1qxZw5YtW6isrGTSpEk0NGinOxFpP4WFl/8i4ufnx6lTp9rs/b73ve9x1113kZaWxuTJk3n//fc5cOAA77333mWfZxjGZaeSzZs3j7KyMveRl5fXZplFpPPr6L5Q2tZxTdO+ImPO7Kq9ce+lf8kn0pKrKkYmJyfz6aefXnD+008/JSkp6ZpDiYh4g27durVJXzh58mS++93v0rdvX/r27csLL7xAaGgo27ZtwzAMFi9ezPz585kyZQppaWmsXLmS6upqVq9eDZxdI+2ll15i7NixDB48mFWrVrF79242b97cZvcrInK+i/1C5ly7du1qsc21vn/Pnj05ePAgADabDafTSUlJSbN2hYWFJCQkXPJ1rFare3R60yEi0lpm94VybTRN+8qM69+4bvMXR0s4VeFoobXIxV1VMfJHP/oR6enpLF++nKNHj3L06FHeeOMNfvrTn16wdpqISGf1gx/8oM37woaGBtasWUNVVRW33HILubm52O32ZuufWa1WRo0axdatW4GW10i7FIfDQXl5ebNDRORKNPU7tbW1F1yrqanhV7/6FZMmTWq39y8qKiIvL8/9JX/IkCH4+/uzadMmd5uCggJycnIYMWJEu+UQka7N7L5Qrk3+mWna3SJVjGyNbpFBDOgWgWHA5n0nzY4jXuqqNrB56qmnKC4uZsaMGTidTgACAwN5+umnmTdvXpsGFBHxVOnp6VRXV7dJX7h7925uueUWamtrCQ0NZe3atfTr189dTDx/RE9CQgJHjx4Frm2NtGefffaKcoqInOtnP/sZr732GkOGDGHWrFlcf/31WCwW9u3bx+9+9zsaGhqYP39+q1+vsrKSQ4cOuR/n5uayc+dOoqOjiY6OJiMjg/vuu4/ExESOHDnCM888Q2xsLP/yL/8CNG4UMW3aNObMmUNMTAzR0dHMnTuXAQMGuHfXFhFpa23dF0rHqXbWc7qy8XN8crSmabfWhP4J7M4vY8MeOw9+u4fZccQLXVUx0mKx8OKLL/KLX/yCffv2ERQURGpqKlarta3ziYh4rLbsC6+//np27txJaWkpf/nLX3j00UfJyspq9l7namn9s9a0mTdvHrNnz3Y/Li8v16YNInJFmnaxvvHGG5k3bx6GYQCNfdaECRN49dVXLzs9+nxffPEFd9xxh/txUx/16KOPsnTpUnbv3s2bb75JaWkpiYmJ3HHHHbz99tuEhYW5n/Pyyy/j5+fH1KlTqampYcyYMaxYsQJfX9+2uGURkQu0dV8oHadpinZ4oB8RQf4mp/Ee30mz8Z8bD7D1UBHltXWEB+rvTq7MVRUjm4SGhnLzzTe3VRYREa/UFn1hQEAAffr0AWDo0KFs376d3/72tzz99NPAhRuHnbv+2blrpJ07OrKwsPCy0xKtVqt+iSQibeJ///d/aWho4NChQxiGQWpq6gWjtVtj9OjR7i/xF7Nhw4YWXyMwMJAlS5awZMmSK35/EZFr0VZ9oXScvOLGKdoaFdnoxIkTrWpnBXpFB3GkuIZ/fF3IPTd1a99g0ulcUzFSRETah2EYOBwOUlJSsNlsbNq0icGDBwPgdDrJysrixRdfBJqvkTZ16lTg7BppCxcuNO0eRKRriYqK0i+pRaTLU1/oXZqKkV1985qaqgqwWBg+fHirn5MwfjqBgyezcc9JFSPliplajFy6dClLly7lyJEjAPTv359f/vKXTJw4EWj8Mv7ss8/y2muvUVJSwrBhw/jd735H//793a/hcDiYO3cuf/rTn9xTcV599VW6d+9uxi2JiFyxZ555hokTJ5KcnExFRQVr1qzho48+Yv369VgsFtLT08nMzCQ1NZXU1FQyMzMJDg7moYceArRGmoiIiIjI1cg7M007Oaprj4x01taAYfCjzDdI7NG7xfblRYUs/fW/kzh4Mv/YX0htXQOB/loORVrP1GJk9+7d+fWvf+2emrhy5UruueceduzYQf/+/Vm4cCGLFi1ixYoV9O3bl+eff55x48axf/9+99pA6enprFu3jjVr1hATE8OcOXOYNGkS2dnZWhtIRLzCyZMn+f73v09BQQEREREMHDiQ9evXM27cOKBx07CamhpmzJjh/sXMxo0btUaaiIiIiMg1OF6iadrnCouOIyo+seWGgLPgAAmh/pysrOPDrwv57oDWPU8ETC5GTp48udnjF154gaVLl7Jt2zb69evH4sWLmT9/PlOmTAEai5UJCQmsXr2a6dOnU1ZWxrJly3jrrbfco39WrVpFcnIymzdvZsKECR1+TyIiV2rZsmWXvW6xWMjIyCAjI+OSbbRGmoiIiIjIlckrPjMyMrprT9O+WmNSI1m94xTrvjqhYqRcER+zAzRpaGhgzZo1VFVVccstt5Cbm4vdbmf8+PHuNlarlVGjRrF161YAsrOzqaura9YmKSmJtLQ0d5uLcTgclJeXNztEREREREREpOvIaxoZ2cWnaV+tMamRAHzwdSEVtXXmhhGvYnoxcvfu3YSGhmK1Wnn88cdZu3Yt/fr1w263A7h3i22SkJDgvma32wkICLhgh7Jz21zMggULiIiIcB/JycltfFciIiIiIiIi4qnKquuoqK0HoLuKkVclNTaQ3nEhOOtdbNp70uw44kVML0Zef/317Ny5k23btvFv//ZvPProo+zdu9d93WKxNGtvGMYF587XUpt58+ZRVlbmPvLy8q7tJkRERERERETEazSNiowNtRIUoHXWr4bFYmHywCQA1n11wuQ04k1ML0YGBATQp08fhg4dyoIFCxg0aBC//e1vsdlsABeMcCwsLHSPlrTZbDidTkpKSi7Z5mKsVivh4eHNDhERERERERHpGpo2r+kepfUir8XkQY3FyE8OnqakymlyGvEWphcjz2cYBg6Hg5SUFGw2G5s2bXJfczqdZGVlMWLECACGDBmCv79/szYFBQXk5OS424iIiIiIiIiInOvs5jWaon0t+sSH0i8xnHqXwXu7C8yO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",
      "text/plain": [
       "<Figure size 1600x1000 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load your dataset, adjust the file path accordingly\n",
    "train = pd.read_csv(\"C:\\\\Users\\\\LENOVO\\\\Downloads\\\\train.csv\")\n",
    "\n",
    "# Assuming 'train' contains the necessary columns for your analysis\n",
    "\n",
    "# Now you can use 'train' in your plotting code\n",
    "fig, axes = plt.subplots(2, 4, figsize=(16, 10))\n",
    "sns.countplot(x='Survived', data=train, ax=axes[0, 0])\n",
    "sns.countplot(x='Pclass', data=train, ax=axes[0, 1])\n",
    "sns.countplot(x='Sex', data=train, ax=axes[0, 2])\n",
    "sns.countplot(x='SibSp', data=train, ax=axes[0, 3])\n",
    "sns.countplot(x='Parch', data=train, ax=axes[1, 0])\n",
    "sns.countplot(x='Embarked', data=train, ax=axes[1, 1])\n",
    "sns.histplot(train['Fare'], kde=True,ax=axes[1,2])\n",
    "sns.histplot(train['Age'].dropna(),kde=True,ax=axes[1,3])\n",
    "\n",
    "# Show the plots\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 742
    },
    "id": "h3yXAMWb2q1F",
    "outputId": "73cdf3ff-b9c5-4dd1-ca73-4b8d64b34da5",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='Fare'>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AAAAKCU/qwNIUvfu3Z0eT548WbNnz9b69evVvHlzi1IBAAAAAAAAKAuPKB//rqCgQO+//75Onjyp2NjYYo/Jz89Xfn6+43FOTo674gEAAAAAAAC4RB7xtWtJ2r59u6pVqya73a7BgwdryZIlatasWbHHJiUlKTg42LFFRES4OS0AAAAAAACAf+Ix5eOVV16prVu3av369Xr00Uc1YMAA7dq1q9hjExISlJ2d7djS09PdnBYAAAAAAADAP/GYr11XrlxZl19+uSQpOjpaqampmjFjht54440ix9rtdtntdndHBAAAAAAAAFACHnPm4/kMw3C6riMAAAAAAACA8sUjznwcM2aMunbtqoiICJ04cUKLFy/W6tWrtWLFCqujAQAAoJyKGv+5fOwBVsdwqV+e62Z1BAAAgBLxiPLx0KFDuu+++5SRkaHg4GC1bNlSK1asUKdOnayOBgAAAAAAAKCUPKJ8nDdvntURAAAAAAAAALiYx17zEQAAAAAAAED5RvkIAAAAAAAAwBSUjwAAAAAAAABMQfkIAAAAAAAAwBSUjwAAAAAAAABMQfkIAB4iMTFRNpvNaQsNDbU6FgC4HfMQAADAe/hZHQAA8JfmzZvriy++cDz29fW1MA0AWId5CAAA4B0oHwHAg/j5+XF2DwCIeQgAAOAt+No1AHiQn376SeHh4YqMjNQ999yjffv2WR0JACzBPAQAAPAOnPkIAB4iJiZGb731lq644godOnRIkyZNUrt27bRz507VqlWr2Nfk5+crPz/f8TgnJ8ddcQHANCWdh8xCAAAAz8WZjwDgIbp27apevXqpRYsW6tixo5YuXSpJSk5OvuBrkpKSFBwc7NgiIiLcFRcATFPSecgsBAAA8FyUjwDgoapWraoWLVrop59+uuAxCQkJys7Odmzp6eluTAgA7vFP85BZCAAA4Ln42jUAeKj8/Hzt3r1bN9544wWPsdvtstvtbkwFAO73T/OQWQgAAOC5OPMRADzEyJEjlZKSov3792vDhg266667lJOTowEDBlgdDQDcinkIAADgPTjzEQA8xMGDB9W3b18dPnxYderU0XXXXaf169erYcOGVkcDALdiHgIAAHgPykcA8BCLFy+2OgIAeATmIQAAgPfga9cAAAAAAAAATEH5CAAAAACAxQYOHKgePXpYHQMAXI7yEQAAAAAAAIApKB8BAAAA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QIWUl5end999VydPnlSnTp3UpEkTqyMBgKkaNmxodQQAFdA333yj++67T1988YUWLVqkX375RQ8++KCaNWumH374wep4AIC/KVP52LNnT/Xr109NmjTRkSNH1LVrV0nS1q1budA4AK83atQonT59WjNmzJAknT59Wtddd5127dqlgIAAjR49WitXrlS7du0sTgoA7pGUlKSQkBCnr11L0ptvvqnff/9dTz75pEXJAHibmJgYbdmyRYMHD1br1q1VWFioSZMmadSoUbLZbFbHAwD8jU9ZXvzyyy/rscceU7NmzbRq1SpVq1ZN0tmvYw8ZMsQlAQHAUy1fvly33HKL4/HChQuVlpamn376SceOHdPdd9+tyZMnW5gQANzrjTfe0FVXXVVkf/PmzfX6669bkAiAN9uzZ49SU1NVv359+fn56X//+59yc3OtjgUAOE+ZysdKlSpp5MiRmjFjhlq1auXYHxcXp4ceeuiS3ycpKUlt2rRRYGCg6tatqx49emjPnj1liQYApktLS1OzZs0cj1euXKm77rpLDRs2lM1m0/Dhw7Vly5ZLfr/Zs2erZcuWCgoKUlBQkGJjY7V8+XIzogOAKTIzMxUWFlZkf506dZSRkXHJ78M8BPBPnnvuOcXGxqpTp07asWOHUlNTtWXLFrVs2VLr1q2zOh4A4G/K9LXrc3bt2qW0tDSdPn3aaf8dd9xxSa9PSUnR0KFD1aZNG505c0Zjx45V586dtWvXLlWtWtUVEQHA5Xx8fJzu7rp+/Xo988wzjsfVq1fXsWPHLvn96tevr+eee85x2Yrk5GTdeeed2rJli5o3b+664ABgkoiICH333XeKjIx02v/dd98pPDz8kt+HeQjgn8yYMUMff/yx49JfzZs31/fff68xY8aoQ4cOys/PtzghAOCcMpWP+/bt07///W9t375dNpvN8R/h566xcak3nVmxYoXT4/nz56tu3bratGmT/vWvf5UlIgCY5qqrrtKnn36q+Ph47dy5U2lpabrpppsczx84cEAhISGX/H7du3d3ejx58mTNnj1b69ev5z+2AZQLDz30kOLi4vTnn3/q5ptvliR9+eWXGj16tJ544olLfh/mIYB/sn37dtWuXdtpX6VKlfTCCy/o9ttvtygVAKA4ZSofhw8frsjISH3xxRdq3Lixvv/+ex05ckRPPPGEXnzxxVK/b3Z2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zWbTiRMniuzPzs7mmj4AvFqPHj2UmZmpunXrqkePHhc8zmazMQ8BVBhXXnml9uzZo0aNGumaa67RG2+8oUaNGun1119XWFiY1fEAeIFbb71VTz31lJ5//nl9/PHHCggI0I033uh4ftu2bbrsssssTAgAOF+Zyscbb7xRSUlJeuedd+Tr6yvp7IXEk5KSdMMNN7gkIAB4osLCwmL/7Cl2TOiioKAgq2MAqGDi4uKUkZEhSRo/fry6dOmihQsXqnLlylqwYIHb8zALAe8zadIk9ezZU+3bt1e1atWUnJysypUrO55/88031blzZwsTAgDOV6by8fnnn1f79u115ZVXOj5t+vbbb5WTk6OvvvrKJQEBwFNt2LBBR48eddxtUZLeeustjR8/XidPnlSPHj00c+ZM2e12C1MCgPlyc3M1atQoffzxx/rzzz+1cuVKvfLKK/rll1/0v//9Tw0aNFDt2rWtjgnAC9SpU0fffvutsrOzVa1aNcdJMOe8//77qlatmkXpAADF8SnLi5s3b65t27apT58+ysrK0okTJ3T//ffrf//7n6KiolyVEQA8UmJiorZt2+Z4vH37dg0aNEgdO3bUU089pU8//VRJSUkWJgQA9xg/frwWLFigbt26qW/fvlq1apUeffRRBQQE6Nprr6V4BOBywcHBRYpH6ez1Zv9+JiQAwHqlOvPx/E+3b7nlFiUnJ/MXSwAVytatW/Xss886Hi9evFgxMTGaO3euJCkiIkLjx49XYmKiRQkBwD0++ugjzZs3T/fcc48kqX///rr++utVUFBQbDkAAACAiqNUZz7+/dPte+65R1988YUeffRRV2cDAI927NgxhYSEOB6npKTo1ltvdTxu06aN0tPTrYgGAG6Vnp7udMOHtm3bys/PT7/99puFqQAAAOAJSnXm4/mfbt977718ug2gwgkJCdH+/fsVERGh06dPa/PmzZowYYLj+RMnTqhSpUoWJgQA9ygoKCjyNUc/Pz+dOXPGokQAAADwFKUqHy/26XZERITLwgGAJ7v11lv11FNP6fnnn9fHH3+sgIAAp9m4bds2XXbZZRYmBAD3MAxDAwcOdLrBVl5engYPHqyqVas69n300UdWxAMAAICFSlU+8uk2AEiTJk1Sz5491b59e1WrVk3JyclOs/HNN99U586dLUwIAO4xYMCAIvvuvfdeC5IAAADA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97Pb7QoKCnLaKjIfHx+NGzeu2OfGjx8vHx+P+NceAAAAAFQYHvNfYb169dJvv/2mTz75RF26dNHq1at17bXXasGCBUWO5UwfAOVRYWGh5s+frwEDBsjP76+rXthsNsXFxWnKlClasmSJduzYoYEDByogIED9+vWzMHH5FB0drRYtWjjta9mypa699lqLEgEAAABAxeUx5aMk+fv7q1OnTho3bpzWrl2rgQMHavz48UWO40wfAOXRF198obS0ND344INFnhs9erTi4uI0ZMgQRUdH69dff9XKlSsVGBhoQdLy79lnn3Wc5ejj46OJEydanAgAAAAAKiaPKh/P16xZM508edLqGADgEp07d5ZhGLriiiuKPGez2ZSYmKiMjAzl5eUpJSVFUVFRFqT0DtWrV1f//v3l4+Oj/v37q3r16lZHAgAAAIAKySPudn3kyBHdfffdevDBB9WyZUsFBgZq48aNmjp1qu68806r4wEAyqFBgwZp0KBBVscAAAAAgArNI8rHatWqKSYmRi+//LL27t2rP//8UxEREXr44Yc1ZswYq+MBAAAAAAAAKAWPKB/tdruSkpKUlJRkdRQAAAAAAAAALuLR13wEAAAAAAAAUH5RPgIAAAAA4AavvfaaIiMj5e/vr9atW+vbb7+96PELFy7U1VdfrYCAAIWFhemBBx7QkSNH3JQWAFyD8hEAAAAAAJO9++67iouL09ixY7VlyxbdeOON6tq1q9LS0oo9fs2aNbr//vs1aNAg7dy5U++//75SU1P10EMPuTk5AJQN5SMAAAAAACabNm2aBg0apIceekhNmzbV9OnTFRERodmzZxd7/Pr169WoUSM9/vjjioyM1A033KBHHnlEGzdudHNyACgbykcAAAAAAEx0+vRpbdq0SZ07d3ba37lzZ61du7bY17Rr104HDx7UsmXLZBiGDh06pA8++EDdunW74M/Jz89XTk6O0wYAVvOIu10DALyfYRjKy8tz28/Kz8+XJNntdtlsNrf8XH9/f7f9LAAAUH4cPnxYBQUFCgkJcdofEhKizMzMYl/Trl07LVy4UH369FFeXp7OnDmjO+64QzNnzrzgz0lKStKECRNcmh0AyoryEQDgFnl5eeratavVMUy1fPlyValSxeoYAADAQ53/IaVhGBf84HLXrl16/PHHNW7cOHXp0kUZGRkaNWqUBg8erHnz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008/6e6779Ytt9yi+++/X9OmTXN1PgDwCKdPn9aePXuczvgGAAAAAAAXd8nlY05OjtNms9n07rvv6vvvv1evXr30zDPPOJ4DAG+Rm5urQYMGKSAgQM2bN1daWpok6fHHH9dzzz1ncToAcK29e/fq6aefVt++fZWVlSVJWrFihXbu3GlxMgAAAJRXl1w+Vq9eXTVq1HDamjZtqoMHD+r1119XjRo1HMcAgLdISEjQDz/8oNWrV8vf39+xv2PHjnr33XctTAYArpWSkqIWLVpow4YN+uijj/THH39IkrZt26bx48dbnA5lMW/ePN18882aN2+e1VHgQfi9AAC4i9+lHvj111+bmQMAPNLHH3+sd999V9ddd51sNptjf7NmzbR3714LkwGAaz311FOaNGmS4uPjFRgY6Nh/0003cU3vcuz48eNauHChCgsLtXDhQvXq1UvVq1e3OhYsdvz4cf33v/+VJP33v//l9wIAYKpLLh/bt28vSTpz5owmT56sBx98UBEREaYFAwBP8Pvvv6tu3bpF9p88edKpjASA8m779u1atGhRkf116tTRkSNHLEgEV3jmmWdUWFgoSSosLNS4ceP0yiuvWJwKVktISHB6PGbMGL322msWpQEAeLsS33DGz89PL774ogoKCszIAwAepU2bNlq6dKnj8bnCce7cuYqNjbUqFgC4XPXq1ZWRkVFk/5YtW1SvXj0LEqGsNm7cqO3btzvt27ZtmzZu3GhRIniCjRs3avfu3U77du3axe8FAMA0pbrb9S233KLVq1e7OAoAeJ6kpCSNHTtWjz76qM6cOaMZM2aoU6dOWrBggSZPnmx1PABwmX79+unJJ59UZmambDabCgsL9d1332nkyJG6//77rY6HEiosLNTEiROLfW7ixImOsyFRsZw7+7U448aN4/cCAGCKS/7a9d917dpVCQkJ2rFjh1q3bq2qVas6PX/HHXe4JBwAWK1du3b67rvv9OKLL+qyyy7TypUrde2112rdunVq0aKF1fEAwGUmT56sgQMHql69ejIMQ82aNVNBQYH69eunp59+2up4KKENGzYoJyen2OdycnK0YcMGzuCvgNatW6fc3Nxin8vNzdW6det0/fXXuzkVAMDblap8fPTRRyVJ06ZNK/KczWYr0Vey09PTlZiYqOXLl+vw4cMKCwtTjx49NG7cONWqVas08QDA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d+9enT9/XsOGDePxPvUwduxYtW7dWpLUpk0bJScnmxwRzPbAAw+obdu2LmXt2rVTYmKiSREFlpUrV6pz586y2WyKjY3Vnj17avW+v//97woJCdE999zj3QABwAvq9cxHAAAA4Lu2b9+u8ePHq0ePHrr77ru1du1a5eXlKTc3V1LVD9dLly7VvHnzNHr0aPXs2VPr16/XxYsXlZmZKUk6d+6cVq9erZdeekn33XefevXqpTfffFOHDh3SX/7yFzOb1yjZbDalpKTIbrdr1qxZstlsZoeEBmD58uUux6+88opJkQSWjRs3aubMmZo3b54OHDiggQMHKikpSXl5edd937lz5/TYY4/pRz/6kY8iBQDPqlPy0WKxVHs+DM+LAQAAgDvnzp2TJLVq1UqSdPz4cRUWFrrMsLJarRo0aJCys7MlSbm5ubp8+bLLOR06dFDPnj2d56BuEhIStHHjRiUkJJgdChoIu92uwYMHS5IGDx4su91ubkABIj09XRMnTtQTTzyhbt26aenSpYqKitKqVauu+75JkyYpOTlZ8fHxPooUADyrThvOGIah8ePHy2q1SpIcDocmT56sZs2aSZLL8yABAAAQuAzDUEpKin7wgx+oZ8+ekqTCwkJJqpbosNvtOnnypPOc0NBQ3XLLLdXOufr+7yorK3P5d2hJSYnH2gH4q9TUVLNDCCjl5eXKzc3VnDlzXMoTExOv+8PK2rVr9cUXX+jNN9/Us88+e8Pr0B8CaIjqlHz87m55jz76aLVzHnvssZuLCAAAAI3etGnT9Mknn2jv3r3V6r67csYwjBuuprneOWlpaVq4cGH9gwUALysqKlJFRYXbH19q+mHls88+05w5c7Rnzx6FhNRu6E5/CKAhqlPyce3atd6KAwD82qpVq7Rq1SqdOHFCktSjRw/Nnz9fSUlJkqoG1QsXLtTrr7+ub775Rv369dOKFSvUo0cPE6MGgPqZPn26tmzZot27d6tjx47O8sjISElVsxvbt2/vLD9z5oxzQB4ZGany8nJ98803LrMfz5w5U+Oy4blz5yolJcV5XFJSoqioKI+2CQA8obY/vlRUVCg5OVkLFy5U165da/359IcAGiI2nAEAH+jYsaOef/557d+/X/v379e9996rESNG6PDhw5Jqt/srADR0hmFo2rRp2rRpkz744AN17tzZpb5z586KjIxUVlaW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",
      "text/plain": [
       "<Figure size 1600x1000 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))\n",
    "train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])\n",
    "train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])\n",
    "train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])\n",
    "train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])\n",
    "train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])\n",
    "sns.boxplot(x=\"Survived\", y=\"Age\", data=train,ax=axesbi[1,1])\n",
    "sns.boxplot(x=\"Survived\", y=\"Fare\", data=train,ax=axesbi[1,2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 607
    },
    "id": "6AlKI1eX2q4C",
    "outputId": "b786f233-3a27-4496-d10d-afc855f9d4b9",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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CA+sQ3hgHSiccbI5ERERkULskSIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghBlLoW7TYarWCKbWIiCiasb1THwMpAK2trTCbzWhtbQ11UYiIiAYM2zv1MZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghXagLQESD65i1C83tdli7epBi0iEtwYB4XRwa2+ywdnUjxaRHRqIB5gRDqItKRBT2GEgRxZCapnYs3ViJ7VVNzm2TCzNw67RCzH9lJzrsDgDAlKIMPH7VGOSmmkJVVCKiiMChPaIYccza5RVEAcC2qkY8/5/DmF9W4Nz28eFG3LthDywd9sEuJhFRRGEgRRQjmtvtXkGUaHtVE0ryUt22fXy4EY1tDKSIolF3d3eoixA1GEgRxQhrV4/fx209vV7bWrv4ZUsUjQ4ePBjqIkQNBlJEMSIl3v+USKPO++sgOV4/UMUhIooKDKSIYkRaogFlhemSj5UWpqOitsVt25SiDGQk8c49IiJ/GEgRxYislHgsv6LYK5iaXJiBRRcWYU15tXPblKIMPHHVGKZAICIKQCMIghDqQoSa1WqF2WyGxWJBSkpKqItDNKDc8kjF65CWeDKPVGtXN5Lj9chIYh4pomgktneNjY1IT5fuoabghLRHatmyZdBoNG5/2dnZzscFQcCyZcuQm5sLk8mECy64APv27XM7hs1mw6JFi5CRkYHExERcdtll+O677wa7KkQRIyslHiNzUnBuwRCMzElBVko8zAkGDM9Mwtn5aRiemcQgiijK6fWc/6iWkA/tnXXWWairq3P+VVZWOh978skn8fTTT2PlypXYuXMnsrOzcdFFF6G1tdW5z+LFi7Fx40asX78e5eXlaGtrw6xZs+BwOEJRHSIiIoohIc9srtPp3HqhRIIg4Nlnn8X999+PK6+8EgDwyiuvICsrC6+//jpuueUWWCwWrF69Gn/9618xY8YMAMCrr76KvLw8fPTRR7jkkksGtS5EREQUW0LeI3X48GHk5uaioKAA1157Lb755hsAQHV1Nerr63HxxRc79zUajZg6dSp27NgBANi1axe6u7vd9snNzcXo0aOd+xARERENlJD2SE2cOBF/+ctfcMYZZ+DYsWN49NFHMWnSJOzbtw/19fUAgKysLLfnZGVl4ciRIwCA+vp6GAwGpKWlee0jPl+KzWaDzWZz/t9qtapVJSIiorDB9m7ghbRHaubMmbjqqqtQXFyMGTNm4P333wfQN4Qn0mg0bs8RBMFrm6dA+6xYsQJms9n5l5eX149aEBERhSe2dwMv5EN7rhITE1FcXIzDhw8750159iw1NDQ4e6mys7Nht9vR3Nzscx8pS5cuhcVicf7V1taqXBMiIqLQY3s38MIqkLLZbDhw4ABycnJQUFCA7OxsbN682fm43W7H1q1bMWnSJADAuHHjoNfr3fapq6vD3r17nftIMRqNSElJcfsjIiKKNmzvBl5I50jdfffd+NGPfoT8/Hw0NDTg0UcfhdVqxdy5c6HRaLB48WIsX74cRUVFKCoqwvLly5GQkIA5c+YAAMxmM2666SbcddddSE9Px5AhQ3D33Xc7hwqJiIiIBlJIA6nvvvsOs2fPRmNjI4YOHYrzzjsPn3zyCU477TQAwD333IPOzk4sWLAAzc3NmDhxIj788EMkJyc7j/HMM89Ap9Ph6quvRmdnJ6ZPn461a9dCq9WGqlpEREQUI7hEDLhEDBERxQa2d+oLqzlSRERERJGEgRQRERGRQgykiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghBlJERERECjGQIiIiIlKIgRQRERGRQgykiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghBlJERERECjGQIiIiIlKIgRQRERGRQgykiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUCptAasWKFdBoNFi8eLFzmyAIWLZsGXJzc2EymXDBBRdg3759bs+z2WxYtGgRMjIykJiYiMsuuwzffffdIJeeiIiIYlFYBFI7d+7Eiy++iDFjxrhtf/LJJ/H0009j5cqV2LlzJ7Kzs3HRRRehtbXVuc/ixYuxceNGrF+/HuXl5Whra8OsWbPgcDgGuxpEREQUY0IeSLW1teG6667DSy+9hLS0NOd2QRDw7LPP4v7778eVV16J0aNH45VXXkFHRwdef/11AIDFYsHq1avx1FNPYcaMGSgpKcGrr76KyspKfPTRR6GqEhEREcWIkAdSt956Ky699FLMmDHDbXt1dTXq6+tx8cUXO7cZjUZMnToVO3bsAADs2rUL3d3dbvvk5uZi9OjRzn2k2Gw2WK1Wtz8iIqJow/Zu4IU0kFq/fj2++OILrFixwuux+vp6AEBWVpbb9qysLOdj9fX1MBgMbj1ZnvtIWbFiBcxms/MvLy+vv1UhIiIKO2zvBl7IAqna2lrcfvvtePXVVxEfH+9zP41G4/Z/QRC8tnkKtM/SpUthsVicf7W1tcEVnoiIKAKwvRt4ulC98K5du9DQ0IBx48Y5tzkcDnz88cdYuXIlDh06BKCv1yknJ8e5T0NDg7OXKjs7G3a7Hc3NzW69Ug0NDZg0aZLP1zYajTAajWpXiYiIKKywvRt4IeuRmj59OiorK7F7927n3/jx43Hddddh9+7dGDZsGLKzs7F582bnc+x2O7Zu3eoMksaNGwe9Xu+2T11dHfbu3es3kCIiIiJSQ8h6pJKTkzF69Gi3bYmJiUhPT3duX7x4MZYvX46ioiIUFRVh+fLlSEhIwJw5cwAAZrMZN910E+666y6kp6djyJAhuPvuu1FcXOw1eZ2IiIhIbSELpOS455570NnZiQULFqC5uRkTJ07Ehx9+iOTkZOc+zzzzDHQ6Ha6++mp0dnZi+vTpWLt2LbRabQhLTkRERLFAIwiCEOpChJrVaoXZbIbFYkFKSkqoi0NERDQg2N6pL+R5pIiIiIgiFQMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghBlJERERECjGQIiIiIlKIgRQRERGRQgykiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFGEgRERERKcRAioiIiEghBlJERERECjGQIiIiIlKIgRQRERGRQgykiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihRhIERERESnEQIqIiIhIIQZSRERERAoxkCIiIiJSiIEUERERkUIMpIiIiIgUYiBFREREpBADKSIiIiKFQhpIvfDCCxgzZgxSUlKQkpKC888/Hx988IHzcUEQsGzZMuTm5sJkMuGCCy7Avn373I5hs9mwaNEiZGRkIDExEZdddhm+++67wa4KERERxaCQBlKnnnoqHn/8cXz++ef4/PPPceGFF+LHP/6xM1h68skn8fTTT2PlypXYuXMnsrOzcdFFF6G1tdV5jMWLF2Pjxo1Yv349ysvL0dbWhlmzZsHhcISqWkRERBQjNIIgCKEuhKshQ4bgd7/7HebPn4/c3FwsXrwYS5YsAdDX+5SVlYUnnngCt9xyCywWC4YOHYq//vWvuOaaawAAR48eRV5eHjZt2oRLLrlE1mtarVaYzWZYLBakpKQMWN2IiIhCie2d+sJmjpTD4cD69evR3t6O888/H9XV1aivr8fFF1/s3MdoNGLq1KnYsWMHAGDXrl3o7u522yc3NxejR4927kNEREQ0UHShLkBlZSXOP/98dHV1ISkpCRs3bsSoUaOcgVBWVpbb/llZWThy5AgAoL6+HgaDAWlpaV771NfX+3xNm80Gm83m/L/ValWrOkRERGGD7d3AC3mP1IgRI7B792588skn+NWvfoW5c+di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VgDhX34A6OnpwQMPPICCggKYTCYMGzYMv/3tb9Hb2+vcJxLqQUESqN/Wr18v6PV64aWXXhL2798v3H777UJiYqJw5MiRUBdN0qZNm4T7779f2LBhgwBA2Lhxo9vjjz/+uJCcnCxs2LBBqKysFK655hohJydHsFqtoSmwhEsuuUR4+eWXhb179wq7d+8WLr30UiE/P19oa2tz7hPu9XjnnXeE999/Xzh06JBw6NAh4b777hP0er2wd+9eQRDCv/yePvvsM+H0008XxowZI9x+++3O7eFej4ceekg466yzhLq6OudfQ0OD8/FwL78gCMLx48eF0047TZg3b57w6aefCtXV1cJHH30kVFVVOfcJ93o0NDS4XYPNmzcLAIT//Oc/giCEf/kFQRAeffRRIT09XXjvvfeE6upq4e9//7uQlJQkPPvss859IqEeFBwGUio499xzhV/+8pdu20aOHCnce++9ISqRfJ6BVG9vr5CdnS08/vjjzm1dXV2C2WwW/u///i8EJZSnoaFBACBs3bpVEITIrUdaWprw5z//OeLK39raKhQVFQmbN28Wpk6d6gykIqEeDz30kDB27FjJxyKh/IIgCEuWLBHKysp8Ph4p9XB1++23C8OHDxd6e3sjpvyXXnqpMH/+fLdtV155pXD99dcLghCZ14EC49BeP9ntduzatQsXX3yx2/aLL74YO3bsCFGplKuurkZ9fb1bfYxGI6ZOnRrW9bFYLACAIUOGAIi8ejgcDqxfvx7t7e04//zzI678t956Ky699FLMmDHDbXuk1OPw4cPIzc1FQUEBrr32WnzzzTcAIqf877zzDsaPH4+f/vSnyMzMRElJCV566SXn45FSD5Hdbserr76K+fPnQ6PRREz5y8rK8O9//xtfffUVAODLL79EeXk5fvjDHwKIvOtA8nCtvX5qbGyEw+FAVlaW2/asrCzU19eHqFTKiWWWqs+RI0dCUaSABEHAnXfeibKyMowePRpA5NSjsrIS559/Prq6upCUlISNGzdi1KhRzi/VcC8/AKxfvx5ffPEFdu7c6fVYJFyHiRMn4i9/+QvOOOMMHDt2DI8++igmTZqEffv2RUT5AeCbb77BCy+8gDvvvBP33XcfPvvsM9x2220wGo244YYbIqYeorfeegstLS2YN28egMh4HwHAkiVLYLFYMHLkSGi1WjgcDjz22GOYPXs2gMipBwWHgZRKPFfRFgQholfWjqT6LFy4EHv27EF5ebnXY+FejxEjRmD37t1oaWnBhg0bMHfuXGzdutX5eLiXv7a2Frfffjs+/PBDxMfH+9wvnOsxc+ZM57+Li4tx/vnnY/jw4XjllVdw3nnnAQjv8gNAb28vxo8fj+XLlwMASkpKsG/fPrzwwgu44YYbnPuFez1Eq1evxsyZM5Gbm+u2PdzL/8Ybb+DVV1/F66+/jrPOOgu7d+/G4sWLkZubi7lz5zr3C/d6UHA4tNdPGRkZ0Gq1Xr1PDQ0NXr86IoF4t1Kk1GfRokV455138J///Aennnqqc3uk1MNgMKCwsBDjx4/HihUrMHbsWPzhD3+ImPLv2rULDQ0NGDduHHQ6HXQ6HbZu3YrnnnsOOp3OWdZwr4erxMREFBcX4/DhwxFzHXJycjBq1Ci3bWeeeSZqamoARM7nAQCOHDmCjz76CDfffLNzW6SU/9e//jXuvfdeXHvttSguLsbPfvYz3HHHHVixYgWAyKkHBYeBVD8ZDAaMGzcOmzdvdtu+efNmTJo0KUSlUq6goADZ2dlu9bHb7di6dWtY1UcQBCxcuBBvvvkmtmzZgoKCArfHI6UengRBgM1mi5jyT58+HZWVldi9e7fzb/z48bjuuuuwe/duDBs2LCLq4cpms+HAgQPIycmJmOtQWlrqlf7jq6++wmmnnQYgsj4PL7/8MjIzM3HppZc6t0VK+Ts6OhAX596sarVaZ/qDSKkHBSk0c9yji5j+YPXq1cL+/fuFxYsXC4mJicK3334b6qJJam1tFSoqKoSKigoBgPD0008LFRUVznQNjz/+uGA2m4U333xTqKysFGbPnh12t+f+6le/Esxms/Df//7X7Zbpjo4O5z7hXo+lS5cKH3/8sVBdXS3s2bNHuO+++4S4uDjhww8/FAQh/Mvvi+tde4IQ/vW46667hP/+97/CN998I3zyySfCrFmzhOTkZOfnN9zLLwh9qSd0Op3w2GOPCYcPHxZee+01ISEhQXj11Ved+0RCPRwOh5Cfny8sWbLE67FIKP/cuXOFU045xZn+4M033xQyMjKEe+65x7lPJNSDgsNASiV//OMfhdNOO00wGAzCOeec47wNPxz95z//EQB4/c2dO1cQhL5bdB966CEhOztbMBqNwpQpU4TKysrQFtqDVPkBCC+//LJzn3Cvx/z5853vmaFDhwrTp093BlGCEP7l98UzkAr3eoh5fPR6vZCbmytceeWVwr59+5yPh3v5Re+++64wevRowWg0CiNHjhRefPFFt8cjoR7/+te/BADCoUOHvB6LhPJbrVbh9ttvF/Lz84X4+Hhh2LBhwv333y/YbDbnPpFQDwqORhAEISRdYUREREQRjnOkiIiIiBRiIEVERESkEAMpIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURB27FjB7RaLX7wgx+EuihERCHFJWKIKGg333wzkpKS8Oc//xn79+9Hfn5+qItERBQS7JEioqC0t7fjb3/7G371q19h1qxZWLt2rdvj77zzDoqKimAymTBt2jS88sor0Gg0aGlpce6zY8cOTJkyBSaTCXl5ebjtttvQ3t4+uBUhIlIBAykiCsobb7yBESNGYMSIEbj++uvx8ssvQ+zY/vbbb/GTn/wEl19+OXbv3o1bbrkF999/v9vzKysrcckll+DKK6/Enj178MYbb6C8vBwLFy4MRXWIiPqFQ3tEFJTS0lJcffXVuP3229HT04OcnBysW7cOM2bMwL333ov3338flZWVzv0feOABPPbYY2hubkZqaipuuOEGmEwm/OlPf3LuU15ejqlTp6K9vR3x8fGhqBYRkSLskSIi2Q4dOoTPPvsM1157LQBAp9PhmmuuwZo1a5yPT5gwwe055557rtv/d+3ahbVr1yIpKcn5d8kll6C3txfV1dWDUxEiIpXoQl0AIoocq1evRk9PD0455RTnNkEQoNfr0dzcDEEQoNFo3J7j2end29uLW265BbfddpvX8TlpnYgiDQMpIpKlp6cHf/nLX/DUU0/h4osvdnvsqquuwmuvvYaRI0di06ZNbo99/vnnbv8/55xzsG/fPhQWFg54mYmIBhrnSBGRLG+99RauueYaNDQ0wGw2uz12//33Y9OmTXjzzTcxYsQI3HHHHbjpppuwe/du3HXXXfjuu+/Q0tICs9mMPXv24LzzzsONN96In//850hMTMSBAwewefNmPP/88yGqHRGRMpwjRUSyrF69GjNmzPAKooC+Hqndu3ejubkZ//jHP/Dmm29izJgxeOGFF5x37RmNRgDAmDFjsHXrVhw+fBiTJ09GSUkJHnzwQeTk5AxqfYiI1MAeKSIaUI899hj+7//+D7W1taEuChGR6jhHiohUtWrVKkyYMAHp6enYvn07fve73zFHFBFFLQZSRKSqw4cP49FHH8Xx48eRn5+Pu+66C0uXLg11sYiIBgSH9oiIiIgU4mRzIiIiIoUYSBEREREpxECKiIiISCEGUkREREQKMZAiIiIiUoiBFBEREZFCDKSIiIiIFGIgRURERKQQAykiIiIihf4/leRPKnRtE60AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 600x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.jointplot(x=\"Age\", y=\"Fare\", data=train);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 682
    },
    "id": "v5jmwOog2q6o",
    "outputId": "2f8b5de5-efbb-496b-dc55-d086c52a9cce",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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GKfPmZSUpLCwsEztYWFhSkpKytE5rl69qlGjRqlr164KCQlx6PpODSqOjY3VnDlzMrXPmTNHEyZMcOaUAAAAwH0vJiZGycnJdltMTEyW+44ZM0YWi+WO283+/llN43+zO/7dXLt2TV26dFFGRoamT5/u8GtyatrRmTNnatGiRZnaH3roIXXp0kUjR4505rQAAADAXblzULEj3YNeeeUVdenS5Y77REVFac+ePfrjjz8yPXf69GmFh4ff8fhr166pU6dOOnr0qNauXetwdUByMiFISkpSREREpvaiRYsqMTHRmVMCAAAAeUqRIkVUpEiRu+5Xv359JScn66efflKdOnUkSVu3blVycrIaNGiQ7XE3k4GDBw/q+++/V+HChZ2K06kuQyVLltTGjRsztW/cuFHFixd3KhAAAADAG1WuXFktW7bUgAEDtGXLFm3ZskUDBgzQ008/bTeguFKlSvrqq68kSdevX1fHjh21bds2LVy4UOnp6UpKSlJSUpLS0tIcur5TFYL+/ftryJAhunbtmp566ilJ0po1azRixAgNHTrUmVMCAAAAOWIo761DsHDhQg0ePFjNmzeXJLVt21YffPCB3T4HDhxQcnKyJOn333/XsmXLJN1YHuBW33//vZ588skcX9uphGDEiBE6d+6cBg0aZMtAAgICNHLkyGwHVdwqq+ma0tLS5O/v70w4AAAAgEcrVKiQPvnkkzvuc+t6YFFRUXaPc8PhLkPp6en68ccfNXLkSJ0+fVpbtmzR7t27de7cOb355ps5OkdsbKxtWeab26ezP3I4eAAAAHgfwzDctuVFDlcIfH191aJFC8XHx6t06dKqXbu2wxeNiYlRdHS0XdvGA785fB4AAAAAueNUl6GqVavqyJEjKl26tFMXzWq6JroLAQAAICcy8uYX9W7j1CxD77zzjoYNG6bly5crMTFRKSkpdhsAAAAAz+BUhaBly5aSbox+vnX1tJurqaWnp5sTHQAAAACXcioh+P77782OAwAAAMiRvDq4112cSggaNWpkdhwAAAAA3MCphOCmK1euKCEhIdNqaNWqVctVUAAAAEB2MqgQmMqphOD06dPq06ePVq5cmeXzjCEAAAAAPINTswwNGTJE58+f15YtWxQYGKhVq1Zp/vz5Kl++vG0JZQAAAAD3P6cqBGvXrtV///tf1a5dWz4+PoqMjFSzZs0U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2FjFxMTIarW6O5w8h/vrWtxf1+L+uhb317W4v67F/cX9ioTACSkpKQoNDVVycrJCQkLcHU6ew/11Le6va3F/XYv761rcX9fi/uJ+RZchAAAAwIuREAAAAABejIQAAAAA8GIkBE6wWq0aPXo0A4JchPvrWtxf1+L+uhb317W4v67F/cX9ikHFAAAAgBejQgAAAAB4MRICAAAAwIuREAAAAABejIQgj+ndu7fat2/v7jBM8cMPP8hisejChQsuvU5eumdZyeuvD3mHxWLR0qVLJUm//fabLBaLdu3a5daYkD3+jYC8475MCHr37i2LxSKLxSI/Pz+VKVNGw4YN0+XLl90dmluNGTNG1atXv+fXPXXqlF588UWVKlVKVqtVxYoVU4sWLbR582aXXrdBgwZKTExUaGioS6/jCXhPuN+mTZvk6+urli1bujsUj3W3z5LExES1atXKoXN++eWXqlu3rkJDQ1WgQAE99NBDGjp0qCvCv+/xOXFv3Xq/b90OHTrk7tAAh+VzdwDZadmypebOnatr165p/fr16t+/vy5fvqwZM2a4O7R7Lj09XRaLxW3Xf+6553Tt2jXNnz9fZcqU0R9//KE1a9bo3LlzTp3PMAylp6crX747//fz9/dXsWLFnLpGXsR7wr3mzJmjv/71r/r444+VkJCgUqVKuTskj3O3zxJH3+/fffedunTpon/84x9q27atLBaL9u/frzVr1rgifI9g1udETj+nvd3N+32rokWLOnSOm7/jfXzuy+9o4SXu2/99N789KlmypLp27apu3bpp6dKl+uSTT1SrVi0VKFBAxYoVU9euXXXq1CnbcefPn1e3bt1UtGhRBQYGqnz58rY3a1paml555RVFREQoICBAUVFRio2NtR2bnJysF154QWFhYQoJCdFTTz2l3bt3256/+Q39v//9b0VFRSk0NFRdunTRxYsXbftcvHhR3bp1U/78+RUREaEpU6boySef1JAhQ2z7pKWlacSIESpRooTy58+vunXr6ocffrA9P2/ePBUsWFDLly9XlSpVZLVadezYsUz3KD09XdHR0SpYsKAKFy6sESNGyOxZZC9cuKANGzZowoQJaty4sSIjI1WnTh3FxMSoTZs2WZaML1y4IIvFYntNN7v+fPvtt6pVq5asVqtmz54ti8WiX3/91e56kydPVlRUlAzDsOsylJycrMDAQK1atcpu/yVLlih//vy6dOmSJOnEiRPq3LmzHnjgARUuXFjt2rXTb7/9dk/vmatk956QpH379qlNmzYKCQlRgQIF1LBhQx0+fDjL86xatUqPP/647R48/fTTdvve7X0yZswY2ze8xYsX1+DBg136uu8Hly9f1meffaaXXnpJTz/9tObNm2f3/LJly1S+fHkFBgaqcePGmj9/fqbubps2bdITTzyhwMBAlSxZUoMHD/aqb27v9lki2XcZuunXX39VgwYNFBAQoIceesjus3L58uV6/PHHNXz4cFWsWFEVKlRQ+/bt9f7779v2ufm5PXPmTJUsWVJBQUF6/vnnXd4V0V2c/d2Z1ef0+vXrlZGRoQkTJqhcuXKyWq0qVaqU3nnnHbtrHjlyRI0bN1ZQUJAeeeQRl1eP7yc37/et23vvvaeqVasqf/78KlmypAYNGmT7HSVl/zv+bn8bAK503yYEtwsMDNS1a9eUlpamt956S7t379bSpUt19OhR9e7d27bf3//+d+3fv18rV65UfHy8ZsyYoSJFikiSpk2bpmXLlumzzz7TgQMH9MknnygqKkrSjW9D2rRpo6SkJK1YsULbt29XjRo11KRJE7tvwg8fPqylS5dq+fLlWr58udatW6fx48fbno+OjtbGjRu1bNkyxcXFaf369dqxY4fda+nTp482btyo//znP9qzZ4+ef/55tWzZUgcPHrTtc+XKFcXGxurjjz/Wvn37FBYWlumeTJo0SXPmzNHs2bO1YcMGnTt3Tl999ZUZt9smODhYwcHBWrp0qVJTU3N1rhEjRig2Nlbx8fHq2LGjatasqYULF9rts2jRInXt2jVTRSQ0NFRt2rTJcv927dopODhYV65cUePGjRUcHKwff/xRGzZsUHBwsFq2bKm0tDRJ9+ae3Ss33xMnTpzQE088oYCAAK1du1bbt29X3759df369SyPu3z5sqKjo/Xzzz9rzZo18vHx0bPPPquMjAxJd36ffPHFF5oyZYpmzpypgwcPaunSpapateq9eslus3jxYlWsWFEVK1ZU9+7dNXfuXFsi+dtvv6ljx45q3769du3apRdffFGvv/663fF79+5VixYt1KFDB+3Zs0eLFy/Whg0b9Morr7jj5biFs58lw4cP19ChQ7Vz5041aNBAbdu21dmzZyXdqCjs27dPv/zyyx3PcejQIX322Wf6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X67z58+rX79+mdbD6Nixo2bPnq0lS5Zo8uTJGjlypPr166ddu3bZusDcfN+PHDlS9erV08svv6wBAwYof/78io+PV1xcnN2MOHnZ3T5LsvOvf/3L9n6dMmWKzp8/r759+0q6MYPQlStX1Lp1a0VGRurChQuaNm2arl27pmbNmtnOERAQoF69eundd99VSkqKBg8erE6dOnnNH1o5+d2ZlYCAAI0cOVIjRoyQv7+/HnvsMZ0+fVr79u1Tv3797kHknqds2bK6fv263n//fT3zzDPauHGjPvzww7seZ+bfBoBT3Dd8IXt3GgS5aNEiIyoqyrBarUb9+vWNZcuW2Q1qfeutt4zKlSsbgYGBRqFChYx27doZR44cMQzDMD766COjevXqRv78+Y2QkBCjSZMmxo4dO2znTklJMf76178axYsXN/z8/IySJUsa3bp1MxISEgzDyDyo1zAMY8qUKUZkZKTdObp27WoEBQUZxYoVMyZPnmzUqVPHGDVqlG2ftLQ048033zSioqIMPz8/o1ixYsazzz5r7NmzxzCMGwOOQkNDM732269/7do149VXXzVCQkKMggULGtHR0UbPnj1NHSB79epVY9SoUUaNGjWM0NBQIygoyKhYsaLxxhtvGFeuXDEMwzD2799v1KtXzwgMDDSqV69urF69OstBxTcHWN7u+eefNyQZc+bMsWvP7rjhw4cbkow333wz07kSExONnj17GkWKFDGsVqtRpkwZY8CAAUZycrJhGPfmnrnC3QYG796922jevLkRFBRkFChQwGjYsKFx+PDhLI+Ni4szKleubFitVqNatWrGDz/8YDeY807vk6+++sqoW7euERISYuTPn9+oV6+e8d1337nqZbvd008/bbRu3TrL57Zv325IMrZv327897//NcqVK2dYrVbjySefNGbMmGFIshsw/NNPPxnNmjUzgoODjfz58xvVqlUz3nnnnXv1UtwuJ58lymJQ8aJFi4y6desa/v7+RuXKlY01a9bYzrl27VrjueeeM0qWLGn4+/sb4eHhRsuWLY3169fb9rn5uTl9+nSjePHiRkBAgNGhQwfj3Llz9/T13wu5+d2Z3edtenq68fbbbxuRkZGGn5+fUapUKeMf//iHYRg5m1QiL8vufk+ePNmIiIgwAgMDjRYtWhgLFiywu7fZ/Y6/298GgCtZDMNDJmH3UJcvX1aJEiU0adIkvlEBvMQ777yjDz/8UMePH3d3KF5vzJgxWrp0qUNdGQHA23hUlyFPsHPnTv3666+qU6eOkpOTNW7cOEm6Y0kcgGebPn26ateurcKFC2vjxo365z//6VVrDAAAPBsJgQu8++67OnDggPz9/VWzZk2tX78+xwOTAXiegwcP6u2339a5c+dUqlQpDR06VDExMe4OCwCAHKHLEAAAAODF7ttpRwEAAAC4HgkBAAAA4MVICAAAAAAvRkIAAAAAeDESAgAAAMCLkRAAAAAAXoyEAAAAAPBiJAQAAACAFyMhAAAAALzY/wOY5jsBqyN/RQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np  # Import NumPy\n",
    "\n",
    "# Load your dataset, adjust the file path accordingly\n",
    "train = pd.read_csv(\"C:\\\\Users\\\\LENOVO\\\\Downloads\\\\train.csv\")\n",
    "\n",
    "# Assuming you want to include only numeric columns in the correlation analysis\n",
    "numeric_columns = train.select_dtypes(include=['float64', 'int64'])\n",
    "\n",
    "f, ax = plt.subplots(figsize=(10, 8))\n",
    "corr = numeric_columns.corr()\n",
    "\n",
    "# Use bool directly instead of np.bool\n",
    "mask = np.zeros_like(corr, dtype=bool)\n",
    "\n",
    "sns.heatmap(corr,\n",
    "            mask=mask,\n",
    "            cmap=sns.diverging_palette(220, 10, as_cmap=True),\n",
    "            square=True, ax=ax)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "id": "bOMJw7b22q82",
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Name_len']=train.Name.str.len()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "4cPRIdja3UzJ",
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Ticket_First']=train.Ticket.str[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "id": "ghCbj7G43U1_",
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['FamilyCount']=train.SibSp+train.Parch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "id": "NluUjHvk3U43",
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Cabin_First']=train.Cabin.str[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "id": "51LBtzlv3U7n",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Regular expression to get the title of the Name\n",
    "train['title'] = train.Name.str.extract('\\, ([A-Z][^ ]*\\.)',expand=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 551
    },
    "id": "mE9Jf5DJ3VA7",
    "outputId": "d52e08b9-3137-484d-bd93-b34a03b275ec",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Mr.</td>\n",
       "      <td>517</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Miss.</td>\n",
       "      <td>182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Mrs.</td>\n",
       "      <td>125</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Master.</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Dr.</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Rev.</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Major.</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Mlle.</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Col.</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Don.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Mme.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Ms.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>Lady.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Sir.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>Capt.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>Jonkheer.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        title  count\n",
       "0         Mr.    517\n",
       "1       Miss.    182\n",
       "2        Mrs.    125\n",
       "3     Master.     40\n",
       "4         Dr.      7\n",
       "5        Rev.      6\n",
       "6      Major.      2\n",
       "7       Mlle.      2\n",
       "8        Col.      2\n",
       "9        Don.      1\n",
       "10       Mme.      1\n",
       "11        Ms.      1\n",
       "12      Lady.      1\n",
       "13       Sir.      1\n",
       "14      Capt.      1\n",
       "15  Jonkheer.      1"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.title.value_counts().reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "HuLJD08s3deH",
    "outputId": "5091a0da-3bf1-41ee-83cd-a4e0d1d05b5d",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "15\n"
     ]
    }
   ],
   "source": [
    "# we see that there are 15 Zero values and its reasonbale\n",
    "# to flag them as missing values since every ticket\n",
    "# should have a value greater than 0\n",
    "print((train.Fare == 0).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "id": "Rb4kHBKo3dhd",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# mark zero values as missing or NaN\n",
    "train.Fare = train.Fare.replace(0, np.NaN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Euo-C2Gb3doq",
    "outputId": "a09e8a9c-904f-45bb-b3ca-7c9b6d48b5a3",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "# validate to see if there are no more zero values\n",
    "print((train.Fare == 0).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "UMzLjwmL3jkR",
    "outputId": "abe7d079-dd80-4ce0-ef38-f42a3f5e4d49",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index([179, 263, 271, 277, 302, 413, 466, 481, 597, 633, 674, 732, 806, 815,\n",
       "       822],\n",
       "      dtype='int64')"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# keep the index\n",
    "train[train.Fare.isnull()].index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "-kSb45S13jnD",
    "outputId": "c6de0d46-b17d-4ec9-8da6-2b4a4554acb1",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "32.75564988584475"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.Fare.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "id": "WdoZmkJL3jsX",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# impute the missing Fare values with the mean Fare value\n",
    "train.Fare.fillna(train.Fare.mean(),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 73
    },
    "id": "1GyjRe1Q3jz3",
    "outputId": "dddb1e1b-44ee-427c-e18f-b2df22d87ae1",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Name_len</th>\n",
       "      <th>Ticket_First</th>\n",
       "      <th>FamilyCount</th>\n",
       "      <th>Cabin_First</th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Name_len, Ticket_First, FamilyCount, Cabin_First, title]\n",
       "Index: []"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# validate if any null values are present after the imputation\n",
    "train[train.Fare.isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "l5WW7xGJ3j2Q",
    "outputId": "7f86bd57-fce3-47b8-ff1f-fe0e97c7e7e2",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "# we see that there are 0 Zero values\n",
    "print((train.Age == 0).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "id": "tO06OHgq3drz",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# impute the missing Age values with the mean Fare value\n",
    "train.Age.fillna(train.Age.mean(),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 73
    },
    "id": "Z8NC0VHc3tJJ",
    "outputId": "9b24a02d-c4aa-423c-d006-cc65d6633c81",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Name_len</th>\n",
       "      <th>Ticket_First</th>\n",
       "      <th>FamilyCount</th>\n",
       "      <th>Cabin_First</th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Name_len, Ticket_First, FamilyCount, Cabin_First, title]\n",
       "Index: []"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# validate if any null values are present after the imputation\n",
    "train[train.Age.isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "utb_dqFG3tPZ",
    "outputId": "7724515c-c86c-494a-a847-893d6f1c3f90",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7710437710437711"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# We see that a majority 77% of the Cabin variable has missing values.\n",
    "# Hence will drop the column from training a machine learnign algorithem\n",
    "train.Cabin.isnull().mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "nPH0QjmG3tSQ",
    "outputId": "d450c726-155f-4a8c-80dc-ffdf2e6ff5cf",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 17 columns):\n",
      " #   Column        Non-Null Count  Dtype  \n",
      "---  ------        --------------  -----  \n",
      " 0   PassengerId   891 non-null    int64  \n",
      " 1   Survived      891 non-null    int64  \n",
      " 2   Pclass        891 non-null    int64  \n",
      " 3   Name          891 non-null    object \n",
      " 4   Sex           891 non-null    object \n",
      " 5   Age           891 non-null    float64\n",
      " 6   SibSp         891 non-null    int64  \n",
      " 7   Parch         891 non-null    int64  \n",
      " 8   Ticket        891 non-null    object \n",
      " 9   Fare          891 non-null    float64\n",
      " 10  Cabin         204 non-null    object \n",
      " 11  Embarked      889 non-null    object \n",
      " 12  Name_len      891 non-null    int64  \n",
      " 13  Ticket_First  891 non-null    object \n",
      " 14  FamilyCount   891 non-null    int64  \n",
      " 15  Cabin_First   204 non-null    object \n",
      " 16  title         890 non-null    object \n",
      "dtypes: float64(2), int64(7), object(8)\n",
      "memory usage: 118.5+ KB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "adnDF13Y3tU2",
    "outputId": "f65f496c-ad1a-4974-c27b-88ea9200b278",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Name_len',\n",
       "       'Ticket_First', 'FamilyCount', 'Cabin_First', 'title'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "id": "BPCWLR9X3tXs",
    "tags": []
   },
   "outputs": [],
   "source": [
    "trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',\n",
    "       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',\n",
    "       'title']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "id": "RMtzIDPg3taT",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# drop rows of missing values\n",
    "trainML = trainML.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ffBUO8S43tcy",
    "outputId": "d5fca4d1-3262-43a0-ad77-62e724535cee",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived        0\n",
       "Pclass          0\n",
       "Name            0\n",
       "Sex             0\n",
       "Age             0\n",
       "SibSp           0\n",
       "Parch           0\n",
       "Ticket          0\n",
       "Fare            0\n",
       "Embarked        0\n",
       "Name_len        0\n",
       "Ticket_First    0\n",
       "FamilyCount     0\n",
       "title           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check the datafram has any missing values\n",
    "trainML.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "id": "NVbRBjIm4AGH",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Import Estimator AND Instantiate estimator class to create an estimator object\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "lr = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "YThPW6um4AJA",
    "outputId": "260e9dc9-840a-468b-ab57-d90c05844dac",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6182432432432432"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_Age = trainML[['Age']].values\n",
    "y = trainML['Survived'].values\n",
    "# Use the fit method to train\n",
    "lr.fit(X_Age,y)\n",
    "# Make a prediction\n",
    "y_predict = lr.predict(X_Age)\n",
    "y_predict[:10]\n",
    "(y == y_predict).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "hGcUnKC14AL8",
    "outputId": "55e50314-69a3-407c-d6d1-3e1cf8519078",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6621621621621622"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_Fare = trainML[['Fare']].values\n",
    "y = trainML['Survived'].values\n",
    "# Use the fit method to train\n",
    "lr.fit(X_Fare,y)\n",
    "# Make a prediction\n",
    "y_predict = lr.predict(X_Fare)\n",
    "y_predict[:10]\n",
    "(y == y_predict).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "8m4z3CQl4AOt",
    "outputId": "8e23674e-df0c-4e15-a749-e57f26b9a05c",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.786036036036036"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_sex = pd.get_dummies(trainML['Sex']).values\n",
    "y = trainML['Survived'].values\n",
    "# Use the fit method to train\n",
    "lr.fit(X_sex, y)\n",
    "# Make a prediction\n",
    "y_predict = lr.predict(X_sex)\n",
    "y_predict[:10]\n",
    "(y == y_predict).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ms6al9604ARW",
    "outputId": "cc6c14bd-aa64-4e22-e72f-6ff932324b82",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6779279279279279"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_pclass = pd.get_dummies(trainML['Pclass']).values\n",
    "y = trainML['Survived'].values\n",
    "lr = LogisticRegression()\n",
    "lr.fit(X_pclass, y)\n",
    "# Make a prediction\n",
    "y_predict = lr.predict(X_pclass)\n",
    "y_predict[:10]\n",
    "(y == y_predict).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "HKSDtemf4AUW",
    "outputId": "36569c60-25d4-422c-afe2-4d9292069bb2",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9887387387387387"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "X=trainML[['Age', 'SibSp', 'Parch',\n",
    "       'Fare', 'Name_len', 'FamilyCount']].values # Taking all the numerical values\n",
    "y = trainML['Survived'].values\n",
    "RF = RandomForestClassifier()\n",
    "RF.fit(X, y)\n",
    "# Make a prediction\n",
    "y_predict = RF.predict(X)\n",
    "y_predict[:10]\n",
    "(y == y_predict).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ADsOnDAU4AWf"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
