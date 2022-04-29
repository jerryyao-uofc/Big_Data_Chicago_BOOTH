### EDA


# Perform EDA on the dataset
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
import pandas as pd
from wordcloud import WordCloud

import chart_studio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


def is_categorical(array_like):
    '''
    Check if a column of data is categorical 
    '''
    return array_like.dtype.name == 'category'

def Get_simple_distribution(df, x):
    '''
    Plot the distribution simple graph
    '''
    title_name = "Distribution of " +x
    sns.displot(data=df, x=x, multiple = "stack").set(title=title_name)
   
def Get_correlation_matrix(df):

    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True).set(title= "Correlation between all features")
    plt.show()

def Get_numeric_visualize(df,x,y):
    '''
       Get density for specific variable x respect to default group and non-default group

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The density plot for specific variable x respect to default group and non-default group
       '''


    # if is_categorical(df[x])==True:
    #     sns.displot(data=df, x="loan_status", hue=x, multiple = "stack")
    # else:
    #     # Without transparency
    #     sns.kdeplot(data=df, x=x, hue="loan_status", cut=0, fill=True, common_norm=False, alpha=0.4)
    title_name = "Distribution of " + x + " in terms of " + y
    sns.kdeplot(data=df, x=x, hue=y, cut=0, fill=True, common_norm=False, alpha=0.4).set(title=title_name)
   


def Get_category_visualize(df,x,y, increase_size = False):
    '''
       Get count plot for specific variable x respect to default group and non-default group

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The count plot for specific variable x respect to default group and non-default group
       '''

    if increase_size == False:
        title_name = "Distribution of " + y + " in terms of " + x
        sns.displot(data=df, x=y, hue=x, multiple = "stack").set(title=title_name)


    title_name = "Distribution of " + y + " in terms of " + x
    p= sns.displot(data=df, x=y, hue=x, multiple = "stack", aspect=1.5).set(title=title_name)
    p.fig.set_dpi(400)
    # plt.show()



def Get_text_visualize(df,x,y = 'loan_status'):
    '''
       Using wordcloud to visualize text data

       Input:
       x     : variable to concern
       y     : Label, loan_status here
       df    : DataFrame, loan_data here

       Output:
       The wordcloud plot for specific variable x respect to default group
       '''


    # drop NAs
    df.dropna(inplace = True)

    df_charged_off = df.loc[df['loan_status'] == 'Charged Off']
    print(df_charged_off.shape)
    string = df_charged_off.emp_title.astype(str)
    string.replace(' ', '_', regex=True)
    string_df = ' '.join(string)
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(string_df)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Charged Off Status: popular job title")
    plt.show()

    df_charged_off = df.loc[df['loan_status'] != 'Charged Off']
    print(df_charged_off.shape)
    string = df_charged_off.emp_title.astype(str)
    string.replace(' ', '_', regex=True)
    string_df = ' '.join(string)
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(string_df)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Fully Paid Status: popular job title")
    plt.show()



def Get_map_visualize(df, x ,y ):
    '''
    def Get_map_visualize(df, x = 'addr_state',y ='loan_status'):

       Using map plot to visualize spatial data

       Input:
       df    :  DataFrame, loan_data here
       x     : 'addr_state', spatial information
       y     :  Label, loan_status here

       Output:
       The map for specific variable x respect to default group
       '''
    

    title_name = "Distribution of " + y + "by state"
    p= sns.displot(data=df, x = 'addr_state',hue =y, multiple = "stack",  height=10, aspect=1.5).set(title=title_name)
    plt.show()
    p.fig.set_dpi(400)


def Get_state_percentage_visulize(df, group_by):
    '''
    Visualize state data for percentage of a given feature
    '''
    (df.groupby('addr_state')[group_by].value_counts(normalize=True).unstack(group_by).plot.bar(stacked=True))
