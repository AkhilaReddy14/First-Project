{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<a href=\"https://www.bigdatauniversity.com\"><img src = \"https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png\" width = 400, align = \"center\"></a>\n",
    "\n",
    "<h1 align=center><font size = 5> Classification with Python</font></h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "In this notebook we try to practice all the classification algorithms that we learned in this course.\n",
    "\n",
    "We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.\n",
    "\n",
    "Lets first load required libraries:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import itertools\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import NullFormatter\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.ticker as ticker\n",
    "from sklearn import preprocessing\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### About dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:\n",
    "\n",
    "| Field          | Description                                                                           |\n",
    "|----------------|---------------------------------------------------------------------------------------|\n",
    "| Loan_status    | Whether a loan is paid off on in collection                                           |\n",
    "| Principal      | Basic principal loan amount at the                                                    |\n",
    "| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |\n",
    "| Effective_date | When the loan got originated and took effects                                         |\n",
    "| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |\n",
    "| Age            | Age of applicant                                                                      |\n",
    "| Education      | Education of applicant                                                                |\n",
    "| Gender         | The gender of applicant                                                               |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets download the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Data From CSV File  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>9/22/2016</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30       9/8/2016   \n",
       "1           2             2     PAIDOFF       1000     30       9/8/2016   \n",
       "2           3             3     PAIDOFF       1000     15       9/8/2016   \n",
       "3           4             4     PAIDOFF       1000     30       9/9/2016   \n",
       "4           6             6     PAIDOFF       1000     30       9/9/2016   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2016   45  High School or Below    male  \n",
       "1  10/7/2016   33              Bechalor  female  \n",
       "2  9/22/2016   27               college    male  \n",
       "3  10/8/2016   28               college  female  \n",
       "4  10/8/2016   29               college    male  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('loan_train.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(346, 10)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Convert to date time object "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0 2016-10-07   45  High School or Below    male  \n",
       "1 2016-10-07   33              Bechalor  female  \n",
       "2 2016-09-22   27               college    male  \n",
       "3 2016-10-08   28               college  female  \n",
       "4 2016-10-08   29               college    male  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['due_date'] = pd.to_datetime(df['due_date'])\n",
    "df['effective_date'] = pd.to_datetime(df['effective_date'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Data visualization and pre-processing\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let’s see how many of each class is in our data set "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PAIDOFF       260\n",
       "COLLECTION     86\n",
       "Name: loan_status, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['loan_status'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "260 people have paid off the loan on time while 86 have gone into collection \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets plot some columns to underestand data better:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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IIcccgjnnHMODzzwQMt2ffr0aXkkxwUXXMD8+fP53Oc+11L3nDlzGDFiBMOGDePwww9nzpw5LesuvPBCBg4cyMaNGwFYu3YtNTU1HfqdtJcft2FmZW3lO29z1HWPFK2+Z7+Vb+zsrUUEZ555JhMnTmwZtbyhoYHVq1dz4YUXMnXqVMaNG8f69es566yzuPPOO1tGiuiIL3zhCy2jnDfbsGEDp5xyCt/97nc59dRTAXjiiSeorq5uGX5p9OjRTJkypWW8vvnz57fs/+KLL3LFFVcwb948amtreeONNzjxxBM54IADGDFiBJA8W2r69OlccsklHY65I3wFZWZWZE888QRVVVVcfPHFLWV1dXUsX76cY489lnHjxgGwyy67cPvtt3PTTTcV7dj33XcfRx99dEtyAvjsZz/b8kDEtkyZMoVrr72W2tpaAGpra7nmmmu49dZbW7aZPHkyt912G5s3by5a3Pk4QZmZFdnSpUs/8UgMyP+ojCFDhvDhhx/ywQcfdPg4ud12dXV1fPTRR60eu73a8ziPwYMHc9xxx/HjH/+408dpD3fxmZl1k20fi5GrtfLtydfFV6h8MeYru/baaznttNM45ZRTinr8XL6CMjMrskMPPZRFixblLd92XMXXX3+dPn360Ldv3y49dkf23zbGfI/zOPDAA6mrq+PBBx/s9LHa4gRlZlZkY8aMYePGjdx9990tZc8//zxDhw5lwYIF/PrXvwaSBxtefvnlXHnllUU79he/+EV+85vf8POf/7yl7JFHHuGll15q1/5XXHEF3/72t2lsbASgsbGRG2+8ka9//euf2PYb3/gGU6ZMKUrc+biLz8zK2oCB+7XrzruO1NcWScyePZvJkydz00030bt3b2pqavje977HQw89xFe/+lUuvfRStmzZwvnnn7/VreUzZszY6rbuZ55JngE7YsQIdtghuaY455xzGDFiBA888MBWz5O68847OeaYY5g7dy6TJ09m8uTJVFVVMWLECL7//e+36/XV1dVx8803c+qpp7Jp0yaqqqq45ZZbWp4QnOvQQw9l1KhR7X50fUe163EbXc2PE7Du4MdtVAY/biM7Cn3cRptdfJKmS3pP0tKcslslvSppiaTZknZPy2skfSSpIZ1+2N5AzMzMcrXnM6gZwLbXx/OAwyJiBLAcuCZn3e8joi6dLsbMzKwT2kxQEfEk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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'Principal', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
     
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "bins = np.linspace(df.age.min(), df.age.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'age', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Pre-processing:  Feature selection/extraction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Lets look at the day of the week people get the loan "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
         "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['dayofweek'] = df['effective_date'].dt.dayofweek\n",
    "bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'dayofweek', bins=bins, ec=\"k\")\n",
    "g.axes[-1].legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below    male          3        0  \n",
       "1 2016-10-07   33              Bechalor  female          3        0  \n",
       "2 2016-09-22   27               college    male          3        0  \n",
       "3 2016-10-08   28               college  female          4        1  \n",
       "4 2016-10-08   29               college    male          4        1  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Convert Categorical features to numerical values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets look at gender:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender  loan_status\n",
       "female  PAIDOFF        0.865385\n",
       "        COLLECTION     0.134615\n",
       "male    PAIDOFF        0.731293\n",
       "        COLLECTION     0.268707\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "86 % of female pay there loans while only 73 % of males pay there loan\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets convert male to 0 and female to 1:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below       0          3        0  \n",
       "1 2016-10-07   33              Bechalor       1          3        0  \n",
       "2 2016-09-22   27               college       0          3        0  \n",
       "3 2016-10-08   28               college       1          4        1  \n",
       "4 2016-10-08   29               college       0          4        1  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## One Hot Encoding  \n",
    "#### How about education?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "education             loan_status\n",
       "Bechalor              PAIDOFF        0.750000\n",
       "                      COLLECTION     0.250000\n",
       "High School or Below  PAIDOFF        0.741722\n",
       "                      COLLECTION     0.258278\n",
       "Master or Above       COLLECTION     0.500000\n",
       "                      PAIDOFF        0.500000\n",
       "college               PAIDOFF        0.765101\n",
       "                      COLLECTION     0.234899\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df.groupby(['education'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Feature befor One Hot Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>education</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>High School or Below</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>Bechalor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender             education\n",
       "0       1000     30   45       0  High School or Below\n",
       "1       1000     30   33       1              Bechalor\n",
       "2       1000     15   27       0               college\n",
       "3       1000     30   28       1               college\n",
       "4       1000     30   29       0               college"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Principal','terms','age','Gender','education']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Feature = df[['Principal','terms','age','Gender','weekend']]\n",
    "Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)\n",
    "Feature.drop(['Master or Above'], axis = 1,inplace=True)\n",
    "Feature.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Feature selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets defind feature sets, X:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = Feature\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "What are our lables?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['loan_status'].values\n",
    "y[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Normalize Data "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Data Standardization give data zero mean and unit variance (technically should be done after train test split )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.51578458,  0.92071769,  2.33152555, -0.42056004, -1.20577805,\n",
       "        -0.38170062,  1.13639374, -0.86968108],\n",
       "       [ 0.51578458,  0.92071769,  0.34170148,  2.37778177, -1.20577805,\n",
       "         2.61985426, -0.87997669, -0.86968108],\n",
       "       [ 0.51578458, -0.95911111, -0.65321055, -0.42056004, -1.20577805,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.48739188,  2.37778177,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.3215732 , -0.42056004,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679]])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X= preprocessing.StandardScaler().fit(X).transform(X)\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Classification "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model\n",
    "You should use the following algorithm:\n",
    "- K Nearest Neighbor(KNN)\n",
    "- Decision Tree\n",
    "- Support Vector Machine\n",
    "- Logistic Regression\n",
    "\n",
    "\n",
    "\n",
    "__ Notice:__ \n",
    "- You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.\n",
    "- You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.\n",
    "- You should include the code of the algorithm in the following cells."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# K Nearest Neighbor(KNN)\n",
    "Notice: You should find the best k to build the model with the best accuracy.  \n",
    "**warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.cross_validation import train_test_split\n",
    "import sklearn.metrics as metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "seed=50\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.40, random_state=seed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 :  0.6906474820143885\n",
      "2 :  0.5827338129496403\n",
      "3 :  0.7266187050359713\n",
      "4 :  0.6762589928057554\n",
      "5 :  0.7482014388489209\n",
      "6 :  0.6762589928057554\n",
      "7 :  0.7482014388489209\n",
      "8 :  0.7050359712230215\n",
      "9 :  0.7769784172661871\n",
      "10 :  0.7338129496402878\n",
      "11 :  0.7482014388489209\n",
      "12 :  0.7194244604316546\n",
      "13 :  0.7482014388489209\n",
      "14 :  0.7266187050359713\n",
      "15 :  0.7410071942446043\n",
      "16 :  0.7338129496402878\n",
      "17 :  0.7482014388489209\n",
      "18 :  0.7266187050359713\n",
      "19 :  0.7553956834532374\n",
      "20 :  0.7266187050359713\n",
      "21 :  0.762589928057554\n",
      "22 :  0.7194244604316546\n",
      "23 :  0.762589928057554\n",
      "24 :  0.7482014388489209\n",
      "25 :  0.762589928057554\n",
      "26 :  0.762589928057554\n",
      "27 :  0.762589928057554\n",
      "28 :  0.762589928057554\n",
      "29 :  0.762589928057554\n",
      "30 :  0.762589928057554\n",
      "31 :  0.7913669064748201\n",
      "32 :  0.7769784172661871\n",
      "33 :  0.7913669064748201\n",
      "34 :  0.7913669064748201\n",
      "35 :  0.7913669064748201\n",
      "36 :  0.7985611510791367\n",
      "37 :  0.7841726618705036\n",
      "38 :  0.7913669064748201\n",
      "39 :  0.7913669064748201\n",
      "40 :  0.7841726618705036\n",
      "41 :  0.8345323741007195\n",
      "42 :  0.8345323741007195\n",
      "43 :  0.8273381294964028\n",
      "44 :  0.8345323741007195\n",
      "45 :  0.8201438848920863\n",
      "46 :  0.8201438848920863\n",
      "47 :  0.8201438848920863\n",
      "48 :  0.8273381294964028\n",
      "49 :  0.8201438848920863\n",
      "50 :  0.8201438848920863\n",
      "51 :  0.8201438848920863\n",
      "52 :  0.8201438848920863\n",
      "53 :  0.8201438848920863\n",
      "54 :  0.8201438848920863\n",
      "55 :  0.8201438848920863\n",
      "56 :  0.8201438848920863\n",
      "57 :  0.8201438848920863\n",
      "58 :  0.8201438848920863\n",
      "59 :  0.8201438848920863\n",
      "60 :  0.8201438848920863\n",
      "61 :  0.8201438848920863\n",
      "62 :  0.8201438848920863\n",
      "63 :  0.8201438848920863\n",
      "64 :  0.8201438848920863\n",
      "65 :  0.8201438848920863\n",
      "66 :  0.8201438848920863\n",
      "67 :  0.8201438848920863\n",
      "68 :  0.8201438848920863\n",
      "69 :  0.8201438848920863\n",
      "70 :  0.8201438848920863\n",
      "71 :  0.8201438848920863\n",
      "72 :  0.8201438848920863\n",
      "73 :  0.8201438848920863\n",
      "74 :  0.8201438848920863\n",
      "75 :  0.8201438848920863\n",
      "76 :  0.8201438848920863\n",
      "77 :  0.8201438848920863\n",
      "78 :  0.8201438848920863\n",
      "79 :  0.8201438848920863\n",
      "80 :  0.8201438848920863\n",
      "81 :  0.8201438848920863\n",
      "82 :  0.8201438848920863\n",
      "83 :  0.8201438848920863\n",
      "84 :  0.8201438848920863\n",
      "85 :  0.8201438848920863\n",
      "86 :  0.8201438848920863\n",
      "87 :  0.8201438848920863\n",
      "88 :  0.8201438848920863\n",
      "89 :  0.8201438848920863\n",
      "90 :  0.8201438848920863\n",
      "91 :  0.8201438848920863\n",
      "92 :  0.8201438848920863\n",
      "93 :  0.8201438848920863\n",
      "94 :  0.8201438848920863\n",
      "95 :  0.8201438848920863\n",
      "96 :  0.8201438848920863\n",
      "97 :  0.8201438848920863\n",
      "98 :  0.8201438848920863\n",
      "99 :  0.8201438848920863\n"
     ]
    }
   ],
   "source": [
    "score=[]\n",
    "for k in range(1,100):\n",
    "    knn=KNeighborsClassifier(n_neighbors=k,weights='uniform')\n",
    "    knn.fit(X_train,y_train)\n",
    "    predKNN=knn.predict(X_test)\n",
    "    accuracy=metrics.accuracy_score(predKNN,y_test)\n",
    "    score.append(accuracy*100)\n",
    "    print (k,': ',accuracy)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "41  :  83.45 %\n"
     ]
    }
   ],
   "source": [
    "print(score.index(max(score))+1,' : ',round(max(score),2),'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "WE can see that the highest accuracy achieved by K=41 as K starts from 1 not zero so the index of k is equal to (score.index(max(score))+1) with accuracy score= 83.45%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0,0.5,'Train Accuracy')"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {

      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(1,100),score)\n",
    "plt.xlabel('Number of Neighbors K')\n",
    "plt.ylabel('Train Accuracy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy :  83.5 %\n"
     ]
    }
   ],
   "source": [
    "knn=KNeighborsClassifier(n_neighbors=41,weights='uniform')\n",
    "knn.fit(X_train,y_train)\n",
    "predKNN=knn.predict(X_test)\n",
    "accuracy=metrics.accuracy_score(predKNN,y_test)\n",
    "print(\"accuracy : \",round(accuracy,3)*100,'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      " COLLECTION       0.67      0.16      0.26        25\n",
      "    PAIDOFF       0.84      0.98      0.91       114\n",
      "\n",
      "avg / total       0.81      0.83      0.79       139\n",
      "\n",
      "\n",
      "\n",
      "Jaccard Similarity Score :  83.45 %\n",
      "\n",
      "\n",
      "F1-SCORE :  [0.25806452 0.90688259]\n",
      "\n",
      "\n",
      "Train Accuracy:  72.46376811594203 %\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report,jaccard_similarity_score,log_loss,f1_score\n",
    "print(classification_report(y_test,predKNN))\n",
    "print('\\n')\n",
    "print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,predKNN)*100,2),'%')\n",
    "print('\\n')\n",
    "print('F1-SCORE : ',f1_score(y_test,predKNN,average=None))\n",
    "print('\\n')\n",
    "print('Train Accuracy: ',metrics.accuracy_score(y_train, knn.predict(X_train))*100,'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\magic\\Anaconda3\\lib\\site-packages\\sklearn\\grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.\n",
      "  DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.grid_search import GridSearchCV\n",
    "dtree=DecisionTreeClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Score: 0.7439613526570048\n",
      "Best params: {'criterion': 'entropy', 'max_depth': 5, 'max_features': 2, 'random_state': 0}\n"
     ]
    }
   ],
   "source": [
    "parameter_grid = {'max_depth': [1, 2, 3, 4, 5,6,5,9,15,20],\n",
    "                  'max_features': [1, 2, 3, 4,5,6,7,8],\n",
    "                 'random_state':[0,15,20,35,50,80,100,150,180,200],\n",
    "                 'criterion':['gini','entropy'],\n",
    "                 }\n",
    "\n",
    "grid_search = GridSearchCV(dtree, param_grid = parameter_grid,\n",
    "                          cv =10)\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "print (\"Best Score: {}\".format(grid_search.best_score_))\n",
    "print (\"Best params: {}\".format(grid_search.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree=DecisionTreeClassifier(max_depth=5,criterion='entropy',max_features=2,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "dtree.fit(X_train,y_train)\n",
    "pred_Dtree=dtree.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      " COLLECTION       0.32      0.40      0.36        25\n",
      "    PAIDOFF       0.86      0.82      0.84       114\n",
      "\n",
      "avg / total       0.76      0.74      0.75       139\n",
      "\n",
      "\n",
      "\n",
      "Jaccard Similarity Score :  74.1 %\n",
      "\n",
      "\n",
      "F1-SCORE :  [0.35714286 0.83783784]\n",
      "\n",
      "\n",
      "Train Accuracy:  77.29468599033817 %\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,pred_Dtree))\n",
    "print('\\n')\n",
    "print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,pred_Dtree)*100,2),'%')\n",
    "print('\\n')\n",
    "print('F1-SCORE : ',f1_score(y_test,pred_Dtree,average=None))\n",
    "print('\\n')\n",
    "print('Train Accuracy: ',metrics.accuracy_score(y_train, dtree.predict(X_train))*100,'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm=SVC().fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_svm=svm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      " COLLECTION       0.33      0.24      0.28        25\n",
      "    PAIDOFF       0.84      0.89      0.87       114\n",
      "\n",
      "avg / total       0.75      0.78      0.76       139\n",
      "\n",
      "\n",
      "\n",
      "Jaccard Similarity Score :  77.7 %\n",
      "\n",
      "\n",
      "F1-SCORE :  [0.27906977 0.86808511]\n",
      "\n",
      "\n",
      "Train Accuracy:  75.84541062801932 %\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,pred_svm))\n",
    "print('\\n')\n",
    "print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,pred_svm)*100,2),'%')\n",
    "print('\\n')\n",
    "print('F1-SCORE : ',f1_score(y_test,pred_svm,average=None))\n",
    "print('\\n')\n",
    "print('Train Accuracy: ',metrics.accuracy_score(y_train, svm.predict(X_train))*100,'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "lgm=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n",
       "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n",
       "          verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lgm.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_lgm=lgm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      " COLLECTION       0.33      0.24      0.28        25\n",
      "    PAIDOFF       0.84      0.89      0.87       114\n",
      "\n",
      "avg / total       0.75      0.78      0.76       139\n",
      "\n",
      "\n",
      "\n",
      "Jaccard Similarity Score :  77.7 %\n",
      "\n",
      "\n",
      "F1-SCORE :  [0.27906977 0.86808511]\n",
      "\n",
      "\n",
      "Train Accuracy:  73.91304347826086 %\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,pred_lgm))\n",
    "print('\\n')\n",
    "print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,pred_lgm)*100,2),'%')\n",
    "print('\\n')\n",
    "print('F1-SCORE : ',f1_score(y_test,pred_lgm,average=None))\n",
    "print('\\n')\n",
    "print('Train Accuracy: ',metrics.accuracy_score(y_train, lgm.predict(X_train))*100,'%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Evaluation using Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import jaccard_similarity_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import log_loss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, download and load the test set:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Test set for evaluation "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2021</td>\n",
       "      <td>10/7/2021</td>\n",
       "      <td>50</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>9/9/2021</td>\n",
       "      <td>9/15/2021</td>\n",
       "      <td>35</td>\n",
       "      <td>Master or Above</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>21</td>\n",
       "      <td>21</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2021</td>\n",
       "      <td>10/9/2021</td>\n",
       "      <td>43</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2021</td>\n",
       "      <td>10/9/2021</td>\n",
       "      <td>26</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>35</td>\n",
       "      <td>35</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>9/11/2016</td>\n",
       "      <td>9/25/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           1             1     PAIDOFF       1000     30       9/8/2021   \n",
       "1           5             5     PAIDOFF        300      7       9/9/2021   \n",
       "2          21            21     PAIDOFF       1000     30      9/10/2021   \n",
       "3          24            24     PAIDOFF       1000     30      9/10/2021   \n",
       "4          35            35     PAIDOFF        800     15      9/11/2021   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2021   50              Bechalor  female  \n",
       "1  9/15/2021   35       Master or Above    male  \n",
       "2  10/9/2021  43  High School or Below  female  \n",
       "3  10/9/2016   26               college    male  \n",
       "4  9/25/2016   29              Bechalor    male  "
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df = pd.read_csv('loan_test.csv')\n",
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>50</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-09-15</td>\n",
       "      <td>35</td>\n",
       "      <td>Master or Above</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>21</td>\n",
       "      <td>21</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-10</td>\n",
       "      <td>2016-10-09</td>\n",
       "      <td>43</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-10</td>\n",
       "      <td>2016-10-09</td>\n",
       "      <td>26</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>35</td>\n",
       "      <td>35</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-11</td>\n",
       "      <td>2016-09-25</td>\n",
       "      <td>29</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           1             1     PAIDOFF       1000     30     2016-09-08   \n",
       "1           5             5     PAIDOFF        300      7     2016-09-09   \n",
       "2          21            21     PAIDOFF       1000     30     2016-09-10   \n",
       "3          24            24     PAIDOFF       1000     30     2016-09-10   \n",
       "4          35            35     PAIDOFF        800     15     2016-09-11   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0 2016-10-07   50              Bechalor  female  \n",
       "1 2016-09-15   35       Master or Above    male  \n",
       "2 2016-10-09   43  High School or Below  female  \n",
       "3 2016-10-09   26               college    male  \n",
       "4 2016-09-25   29              Bechalor    male  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df['due_date'] = pd.to_datetime(test_df['due_date'])\n",
    "test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])\n",
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
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
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>50</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-09-15</td>\n",
       "      <td>35</td>\n",
       "      <td>Master or Above</td>\n",
       "      <td>male</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>21</td>\n",
       "      <td>21</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-10</td>\n",
       "      <td>2016-10-09</td>\n",
       "      <td>43</td>\n",
