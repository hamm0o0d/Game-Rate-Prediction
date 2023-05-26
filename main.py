import streamlit as sl
import joblib
from datetime import date, datetime
import pandas as pd
import numpy as np  # for numerical operations
import pickle
import category_encoders as ce
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_classif
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, PolynomialFeatures, StandardScaler, LabelEncoder, \
    OrdinalEncoder

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))


def calculate_name_words_number(data):
    y = data['Name'].str.replace('[^\w\s]', '').str.split(':| ').apply(lambda x: [str(s) for s in x])
    listy = []
    for item in y:
        listy.append(len(item) + 1)
    return listy


def calculate_subtitle_words_number(data):
    mydata = data['Subtitle'].fillna('one')
    y = mydata.str.replace('[^\w\s]', '', regex=True).str.split(':| ').apply(lambda x: [str(s) for s in x])
    listy = []
    one = ['one']
    for item in y:
        if item == one:
            listy.append(len(item))
        else:
            listy.append(len(item) + 1)
    return listy


def has_price_feature(data):
    lsty = []
    for item in data['Price']:
        if item == 0:
            lsty.append(0)
        else:
            lsty.append(1)
    return lsty


def encode_rate(data):
    return data['Rate'].replace('Low', 0).replace('Intermediate', 1).replace('High', 2)


def encode_subtitle(data):
    listy = []
    mydata = data['Subtitle'].fillna(-11)
    for row in mydata:
        if row == -11:
            listy.append(0)
        else:
            listy.append(1)

    return listy


def encode_age_rating(data):
    data['Age Rating'] = data['Age Rating'].apply(lambda x: x.replace("+", ""))
    data['Age Rating'] = data['Age Rating'].apply(lambda x: int(x))
    return data['Age Rating']


def check_outliar(data):
    # Calculate the interquartile range (IQR) of the data
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Print the outliers
    return (len(outliers))


def handle_outlier(data):
    # Create a sample dataset
    thedata = data

    # Set the Winsorization percentiles
    pct = 0.20

    # Calculate the percentile values
    low_val = np.percentile(thedata, pct * 100)
    high_val = np.percentile(thedata, (1 - pct) * 100)

    # Replace the extreme values with the percentile values
    thedata[data < low_val] = low_val
    thedata[data > high_val] = high_val

    # Print the transformed data
    # print(data)
    data = thedata


def purchases_or_not_feature(data):
    lsty = []
    for item in data['In-app Purchases']:
        if item == '0.0':
            lsty.append(0)
        else:
            lsty.append(1)
    return lsty


def in_app_purchase_eng(data):
    data['In-app Purchases'] = data['In-app Purchases'].str.split(' ').apply(lambda x: [float(s) for s in x if s != ''])
    meanc = []
    medianc = []
    maxc = []
    minc = []
    for row in data['In-app Purchases']:
        meanc.append(np.mean(row))
        medianc.append(np.median(row))
        maxc.append(np.max(row))
        minc.append(np.min(row))
    return meanc, medianc, maxc, minc


def drop_col(data):
    droped = ['Description', 'Developer', 'ID', 'In-app Purchases', 'Subtitle', 'URL', 'Languages', 'Icon URL',
              'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date']
    data.drop(columns=droped, inplace=True)


def convert_date_to_numeric_timestamp(dates):
    timestamps = []
    for date in dates:
        dt_obj = datetime.strptime(date, '%d/%m/%Y')
        timestamp = datetime.timestamp(dt_obj)
        timestamps.append(timestamp)
    return timestamps


def convert_date(data):
    data = data.apply(pd.Timestamp)
    day = data.dt.day
    month = data.dt.month
    year = data.dt.year

    return day, month, year


def calculate_app_ages(data):
    cday, cmonth, cyear = convert_date(data['Current Version Release Date'])
    oday, omonth, oyear = convert_date(data['Original Release Date'])
    cday = list(cday)
    cmonth = list(cmonth)
    cyear = list(cyear)
    oday = list(oday)
    omonth = list(omonth)
    oyear = list(oyear)

    days = []
    months = []
    years = []
    testc, testo = 0, 0
    for i in range(len(cyear)):
        tmp_day = ((datetime(cyear[i], cmonth[i], cday[i])) - (datetime(oyear[i], omonth[i], oday[i]))).days
        if tmp_day < 0:
            tmp_day = tmp_day * (- 1.0)
        tmp_month = tmp_day / 30.5
        tmp_month = "{:.{prec}f}".format(tmp_month, prec=3)
        tmp_month = float(tmp_month)
        if tmp_month < 0:
            tmp_month = tmp_month * (- 1.0)

        tmp_year = tmp_day / 365.25
        tmp_year = "{:.{prec}f}".format(tmp_year, prec=3)
        tmp_year = float(tmp_year)
        if tmp_year < 0:
            tmp_year = tmp_year * (- 1.0)

        days.append(tmp_day)
        months.append(tmp_month)
        years.append(tmp_year)

    return years, months, days


def dev_app_count_col(data):
    # filename = "developer_apps_count.joblib"
    # developer_apps_count=joblib.load(filename)
    developer_apps_count = pd.read_pickle('C:/Users/ahmed/PycharmProjects/deployment/developer_apps_count.pkl')
    listo = []
    for dev in data['Developer']:

        try:
            listo.append(developer_apps_count.loc[dev, 'App Count'])
        except:
            listo.append(1)

    return listo


def calc_density(data):
    data['density'] = data['User Rating Count'] / data['Size']


def description_features(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    punc = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

    Preprocessed_Description_before_stopwords = []
    Preprocessed_Description_after_stopwords = []

    num_of_words_before_remove_stopwords = []
    num_of_words_after_remove_stopwords = []

    percentage_list = []
    lemmatizer = WordNetLemmatizer()
    for index, row in data.iterrows():
        sentence = row['Description']
        # cleaning the Descriptions from newline char and the punctuation and links and the numbers and garbadge data
        sentence = sentence.replace("\\n", "\n")
        sentence = sentence.replace("\n", " ")
        sentence = re.sub(r'u[\da-f]{4}', '', sentence)
        sentence = re.sub(r'http\S+', '', sentence)
        sentence = re.sub(r'\d+', '', sentence)
        sentence_without_punc = ''.join([ele for ele in sentence if ele not in punc])
        # Apply the Word Tokanization
        word_tokens = word_tokenize(sentence_without_punc)
        # Removing the . char from the Tokens
        for item in word_tokens:
            if '.' in item:
                word_tokens.remove(item)

        # Count the number of words before removing the stopwords
        num_of_words_before_remove_stopwords.append(len(word_tokens))

        # Append All Preprocessed tokens before removing stop words in a new list called Preprocessed_Description
        Preprocessed_Description_before_stopwords.append(word_tokens)

        # Removing the StopWords from tokens
        tokens_without_stopwords = [word for word in word_tokens if word.casefold() not in stop_words]

        # Count the number of words after removing the stopwords
        num_of_words_after_remove_stopwords.append(len(tokens_without_stopwords))

        # Applying the Lemmatizations on the tokens
        lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in tokens_without_stopwords]

        # Append All Preprocessed tokens after removing stop words in a new list called Preprocessed_Description
        Preprocessed_Description_after_stopwords.append(lemmatized_words)
    # Putting the defualt value (1) as this description does not have any words and measure the precentage between the number of words after removing the stopwords and between the words befor removing the stopwords
    for i in range(0, len(num_of_words_after_remove_stopwords)):
        if (num_of_words_after_remove_stopwords[i] == 0 or num_of_words_before_remove_stopwords[i] == 0):
            num_of_words_after_remove_stopwords[i] = num_of_words_after_remove_stopwords[i] + 1
            num_of_words_before_remove_stopwords[i] = num_of_words_before_remove_stopwords[i] + 1
        percentage_list.append((num_of_words_after_remove_stopwords[i] / num_of_words_before_remove_stopwords[i]))
    # data.loc[index,'Description']=preprocessing_Des
    data['length_of_all_discription'] = num_of_words_before_remove_stopwords
    data['length_of_discription_summary'] = num_of_words_after_remove_stopwords
    data['discription_summary_percent'] = percentage_list


def Cleaning_Descriptions(sentence):
    punc = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
    sentence = sentence.replace("\\n", "\n")
    sentence = sentence.replace("\n", " ")
    sentence = re.sub(r'u[\da-f]{4}', '', sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub(r'\d+', '', sentence)
    return ''.join([ele for ele in sentence if ele not in punc])


def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    return ' '.join([word for word in word_tokens if word not in stop_words])


def remove_additional_space(text):
    text = re.sub('\s+', ' ', text)
    return text


def Summarization(text):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict()
    for word in words:
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    average = int(sumValues / (len(sentenceValue) + 1))
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    return summary


stemmer = PorterStemmer()


def stemm_words(text):
    word_tokens = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in word_tokens])


lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}


def lemmatize_words(text):
    word_tokens = word_tokenize(text)
    pos_text = pos_tag(word_tokens)
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])


def fill_nulls(data):
    org_rd = '2/9/2016'
    curr_rd = '31/07/2019'

    if data['Name'].isna().sum() > 0:  # mode of xtrain word number
        data['Name'].fillna('a c g', inplace=True)

    # subtitle -> filled in another function

    if data['User Rating Count'].isna().sum() > 0:  # mode of xtrain
        data['User Rating Count'].fillna(5, inplace=True)

    if data['Price'].isna().sum() > 0:  # mode of xtrain
        data['Price'].fillna(0.00, inplace=True)

    if data['In-app Purchases'].isna().sum() > 0:  # this means no purchases
        data['In-app Purchases'].fillna('0.0', inplace=True)

    if data['Description'].isna().sum() > 0:
        data['Description'].fillna('happy', inplace=True)

    if data['Developer'].isna().sum() > 0:  # mode of xtrain
        data['Developer'].fillna('Tapps Tecnologia da Informa\xe7\xe3o Ltda.', inplace=True)

    if data['Age Rating'].isna().sum() > 0:  # mode of xtrain
        data['Age Rating'].fillna('4+', inplace=True)

    if data['Languages'].isna().sum() > 0:
        data['Languages'].fillna('EN', inplace=True)

    if data['Size'].isna().sum() > 0:  # median of xtrain
        data['Size'].fillna(67060736.0, inplace=True)

    if data['Primary Genre'].isna().sum() > 0:  # mode of xtrain
        data['Primary Genre'].fillna('Games', inplace=True)

    if data['Genres'].isna().sum() > 0:  # mode of xtrain
        data['Genres'].fillna('Games, Strategy, Puzzle, Simulation, Action', inplace=True)

    if data['Original Release Date'].isna().sum() > 0:  # mode of xtrain
        data['Original Release Date'].fillna(org_rd, inplace=True)

    if data['Current Version Release Date'].isna().sum() > 0:  # mode of xtrain
        data['Current Version Release Date'].fillna(curr_rd, inplace=True)


def prepare_data(name, subtitle, User_Rating_Count, size, curr_date, orig_date, genere, language, developers,
                 in_app_purchases, description):
    org_rd = '2/9/2016'
    curr_rd = '31/07/2019'

    if len(name) == 0:  # mode of xtrain word number
        name = 'a c g'
    subtitle=str(subtitle)
    if len(subtitle) == 0:
        subtitle = 'one'
    if User_Rating_Count == 0:  # mode of xtrain
        User_Rating_Count = 5

    if size == 0:  # median of xtrain
        size = 67060736.0

    curr_date = date.today()

    orig_date = date.today()
    if curr_date == date.today():
        curr_date = curr_rd
    else:
        curr_date=str(curr_date)

    if orig_date == date.today():  # mode of xtrain
        orig_date = org_rd
    else:
        orig_date = str(orig_date)
    if len(genere) == 0:  # this means no purchases
        genere = 'Games'
    else:
        s = ''

        for i in genere:
            s += i+ ' '
            genere = s

    if len(language) == 0:  # this means no purchases
        language = 'EN'
    else:
        s = ''
        for i in language:
            s += i+ ' '
            language = s

    if len(developers) == 0:  # mode of xtrain
        developers = 'Tapps Tecnologia da Informa\xe7\xe3o Ltda.'

    if len(in_app_purchases) == 0:  # this means no purchases
        in_app_purchases = '0.0'
    else:
        s = ''
        for i in in_app_purchases:
            s += i+ ' '
        in_app_purchases = s

    if len(description) == 0:
        description = 'happy'

    return name, subtitle, User_Rating_Count, size, curr_date, orig_date, genere, language, developers, in_app_purchases, description


lang_mlb = joblib.load("lang_mlb.joblib")
Genres_mlb = joblib.load("Genres_mlb.joblib")
cat_encoder = joblib.load("cat_encoder.joblib")
target_column = joblib.load("target_column.joblib")

terget_encoder = pd.read_pickle("target_enc.pkl")
cv = joblib.load("count_vectorizer.joblib")
selected_columns = joblib.load("selected_columns.joblib")
standard_scaler = pd.read_pickle("standardization.pkl")

# models

# lr=joblib.load( "C:/Users/ahmed/PycharmProjects/deployment/logestic.joblib")
# lda=pd.read_pickle('C:/Users/ahmed/PycharmProjects/deployment/LinearDiscriminantAnalysis.pkl')
# knn=joblib.load('C:/Users/ahmed/PycharmProjects/deployment/knn.joblib')
rf = pd.read_pickle('RF.pkl')



sl.title('Welcome to Game Rate prediction model')
with sl.form('fill the required data to predict the game rate'):
    name, id = sl.columns(2)
    name = sl.text_input('Name')
    id = sl.number_input('ID', value=0, step=1, min_value=0, format="%d")
    url = sl.text_input('Game URL')
    subtitle = sl.text_input('Subtitle')
    icon_url = sl.text_input('Icon URL')
    Price, User_Rating_Count, Age_Rating = sl.columns(3)
    price = sl.number_input('Price', value=0, step=1, min_value=0, format="%d")
    User_Rating_Count = sl.number_input('User Rating Count', value=0, step=1, min_value=0, format="%d")
    Age_Rating = sl.selectbox('Age Rating', options=('4+', '9+', '12+', '17+'))
    size = sl.number_input('Size (in Byte)', value=0, step=1, min_value=0, format="%d")
    primary_genre = sl.selectbox('Primary Genre', options=('Games', 'Sports', 'Utilities', 'Education', 'Entertainmet',
                                                           'Music', 'Business', 'Social Networking', 'Book',
                                                           'Productivity',
                                                           'Finance', 'Reference', 'Stickers', 'Health & Fitness',
                                                           'News',
                                                           'Lifestyle', 'Medical', 'Food & Drink ', 'Shopping'))
    orig_date = sl.date_input('Original Release Date')
    curr_date = sl.date_input('Current Version Release Date')
    genere = sl.multiselect('Genere', options=('Action', 'Adventure', 'Board', 'Books', 'Business', 'Card', 'Casino',
                                               'Casual', 'Education', 'Entertainment', 'Family', 'Finance',
                                               'Food & Drink', 'Games', 'Gaming', 'Health & Fitness', 'Kids & Family',
                                               'Lifestyle', 'Medical', 'Music', 'Navigation', 'News', 'Photo & Video',
                                               'Productivity', 'Puzzle', 'Racing', 'Reference', 'Role Playing',
                                               'Shopping', 'Simulation', 'Social Networking', 'Sports', 'Stickers',
                                               'Strategy', 'Travel', 'Trivia', 'Utilities', 'Word'))
    language = sl.multiselect('Languages', options=('AF', 'AM', 'AR', 'AS', 'AY', 'AZ', 'BE', 'BG', 'BN', 'BO', 'BR',
                                                    'BS', 'CA', 'CS', 'CY', 'DA', 'DE', 'DZ', 'EL', 'EN', 'EO', 'ES',
                                                    'ET', 'EU', 'FA', 'FI', 'FO', 'FR', 'GA', 'GD', 'GL', 'GN', 'GU',
                                                    'GV',
                                                    'HE', 'HI', 'HR', 'HU', 'HY', 'ID', 'IS', 'IT', 'IU', 'JA', 'JV',
                                                    'KA',
                                                    'KK', 'KL', 'KM', 'KN', 'KO', 'KR', 'KS', 'KU', 'KY', 'LA', 'LO',
                                                    'LT',
                                                    'LV', 'MG', 'MK', 'ML', 'MN', 'MR', 'MS', 'MT', 'MY', 'NB', 'NE',
                                                    'NL',
                                                    'NN', 'NO', 'OM', 'OR', 'PA', 'PL', 'PS', 'PT', 'QU', 'RN', 'RO',
                                                    'RU', 'RW', 'SA', 'SD', 'SE', 'SI', 'SK', 'SL', 'SO', 'SQ', 'SR',
                                                    'SU', 'SV', 'SW', 'TA', 'TE', 'TG', 'TH', 'TI', 'TK', 'TL', 'TO',
                                                    'TR', 'TT', 'UG', 'UK', 'UR', 'UZ', 'VI', 'YI', 'ZH'))

    in_app_purchases = sl.multiselect('In-App-Purchases',
                                      options=(' 0.0', ' 0.99', ' 1.49', ' 1.99', ' 10.99', ' 109.99',
                                               ' 11.99', ' 12.99', ' 13.99', ' 139.99', ' 14.99', ' 15.99',
                                               ' 16.99', ' 169.99', ' 17.99', ' 18.99', ' 19.49', ' 19.99',
                                               ' 199.99', ' 2.49', ' 2.99', ' 20.99', ' 21.99', ' 22.99',
                                               ' 23.49', ' 23.99', ' 24.99', ' 25.99', ' 26.99', ' 27.99',
                                               ' 28.99', ' 29.99', ' 3.49', ' 3.99', ' 30.99', ' 31.99',
                                               ' 32.99', ' 33.99', ' 34.99', ' 35.99', ' 36.99', ' 37.99',
                                               ' 38.99', ' 39.99', ' 4.49', ' 4.99', ' 40.99', ' 43.99',
                                               ' 44.99', ' 45.99', ' 46.99', ' 47.99', ' 48.99', ' 49.99',
                                               ' 5.49', ' 5.99', ' 54.99', ' 59.99', ' 6.99', ' 64.99',
                                               ' 69.99', ' 7.99', ' 74.99', ' 79.99', ' 8.49', ' 8.99', ' 84.99',
                                               ' 89.99', ' 9.49', ' 9.99', ' 94.99', ' 99.99', '0', '0.0', '0.99',
                                               '1.49', '1.99', '10.99', '11.99', '119.99', '12.99', '13.99', '14.99',
                                               '15.99', '16.99', '17.99', '19.99', '2.49', '2.99', '21.99', '23.99',
                                               '24.99', '25.99', '29.99', '3.99', '31.99', '35.99', '37.99', '39.99',
                                               '4.99', '49.99', '5.99', '54.99', '59.99', '6.99', '7.49', '7.99',
                                               '79.99', '8.99', '89.99', '9.99', '99.99'))
    developers = sl.selectbox('Developer', options=('Don\'t Blink Studios', 'Ellie\'s Games, LLC',
                                                    'Igor\'s Software Labs LLC', 'Lion\'s Den',
                                                    '"Raffaele D\'Amato"', '"Sean O\'Connor"', 'Weeny Brain\'s Game',
                                                    '1 Simple Game', '108km Tech Ltd', '10K BULBS LLC',
                                                    '11 bit studios s.a.', '111 (LLC)', '111%', '12 POINT APPS LLC',
                                                    '1791 Entertainment LLC', '1C Mobile Ltd', '2Gear', '2K', '2ka',
                                                    '305 Games', '31x Limited', '38 Softworks Inc.', '3909',
                                                    '3D Avenue', '3g60', '3way Interactive', '4s-games',
                                                    '50 Caliber Mobile Inc.', '505 Games (US), Inc.',
                                                    '51st Parallel Ltd', '57Digital Ltd', '5mina', '5minlab Co., Ltd.',
                                                    '5th Planet Games Development ApS', '7 Pirates Limited',
                                                    '8 x 8 Media AG', '8Floor', '9 FACTORY', '99Games', '99bosses',
                                                    'A Brainy Choice, Inc.', 'A Dark Matter Creation LLC',
                                                    'A L Fernando', 'A S K products', 'A Sharp, LLC',
                                                    'A Thinking Ape Entertainment Ltd.', 'A Trillion Games Ltd',
                                                    'A&E Television Networks Mobile', 'A.R.T. Games Co., Ltd',
                                                    'ABIGAMES PTE. LTD', 'ACLAP', 'AE Mobile', 'AE Mobile Inc.',
                                                    'AFEEL, Inc.', 'AFKSoft', 'AISU Technologies, LLC', 'AKPublish',
                                                    'ALBCOM, LLC', 'ALEXEY OSTROGRADSKIY', 'ALSEDI Group',
                                                    'AMT Games Inc.', 'AMT Games Publishing Ltd.', 'AMZN Mobile LLC',
                                                    'ANGames', 'APPDEKO SIA', 'APPcalyptus UG', 'ARAPPDEV',
                                                    'ARTE Experience', 'ARTHUR MIRZOYAN',
                                                    'ASOCIACION ESPA\\xd1OLA DE BRIDGE', 'ASSIST ENTERTAINMENT, K.K',
                                                    'ASSIST Software SRL', 'ATTOMEDIA Corp.', 'AZ man',
                                                    'Aakash Thumaty', 'Aaron Steed', 'Abbacore LLC', 'Abdala tawfik',
                                                    'Abdullah AlAzemi', 'Abel Galvan', 'Abele Games',
                                                    'Abhishek Malpani', 'Abnormal Head LLC (CA)', 'Absolutist Ltd',
                                                    'Accidental Fish Ltd.', 'AceViral.com', 'Acram Digital', 'Actop',
                                                    'AdTeam', 'Adalgisa Raniolo', 'Adam Alad', 'Adam Hensel',
                                                    'Adam Irvine', ..., 'zhe zhang',
                                                    'zhiguo lin', 'zhijuan zhang'))
    description = sl.text_area('Description')
    stat = sl.form_submit_button('view game rate')
    if stat:
        print((curr_date))
        name, subtitle, User_Rating_Count, size, curr_date, orig_date, genere, language, developers, in_app_purchases, description = prepare_data(
            name, subtitle, User_Rating_Count, size, curr_date, orig_date, genere, language, developers,
            in_app_purchases, description)
        data = pd.DataFrame({'URL': [url], 'ID': [id], 'Name': [name], 'Subtitle': [subtitle], 'Icon URL': [icon_url],
                             'User Rating Count': [User_Rating_Count], 'Price': [price],
                             'In-app Purchases': [in_app_purchases], 'Description': [description],
                             'Developer': [developers], 'Age Rating': [Age_Rating], 'Languages': [language],
                             'Size': [size], 'Primary Genre': [primary_genre], 'Genres': [genere],
                             'Original Release Date': [orig_date], 'Current Version Release Date': [curr_date]})
        data['has_purchases'] = purchases_or_not_feature(data)
        data['Age Rating']=encode_age_rating(data)
        data['Name']=calculate_name_words_number(data)
        data['Curr_datestamp']=convert_date_to_numeric_timestamp(data['Current Version Release Date'])
        data['orig_datestamp']=convert_date_to_numeric_timestamp(data['Original Release Date'])
        print(data['In-app Purchases'])
        data['mean Purchases'],data['meedian Purchases'],data['max Purchases'],data['min Purchases']=in_app_purchase_eng(data)
        calc_density(data)
        data["orgDay"],data["orgMonth"],data["orgYear"]=convert_date(data["Original Release Date"])
        data["currDay"],data["currMonth"],data["currYear"]=convert_date(data["Current Version Release Date"])
        data['app age in years'],data['app age in months'],data['app age in days']=calculate_app_ages(data)
        data['has_price']=has_price_feature(data)
        data['has_subtitle']=encode_subtitle(data)
        data['subtitle_words_number']=calculate_subtitle_words_number(data)

        # # number of app for developers to xtrain and xtest
        data['dev_apps_count']=dev_app_count_col(data)
        data["Languages"]=data["Languages"].str.split(' ')
        data["Genres"]=data["Genres"].str.split(' ')

        expandedLabelData_lang = lang_mlb.transform(data.Languages)
        languages_df = pd.DataFrame(expandedLabelData_lang, columns=lang_mlb.classes_)

        expandedLabelData_genres = Genres_mlb.transform(data.Genres)
        genres_df = pd.DataFrame(expandedLabelData_genres, columns=Genres_mlb.classes_)


        # fit and transform the "Primary Genre" column
        ohe_Primary_Genre = cat_encoder.transform(data[['Primary Genre']])
        # create a new DataFrame with the one-hot encoded columns
        encoded_PrimaryGenre = pd.DataFrame(ohe_Primary_Genre.toarray(), columns=cat_encoder.categories_)
        # df_encoded = df_encoded.reset_index(level=1)
        encoded_PrimaryGenre = encoded_PrimaryGenre.reset_index(level=0, drop=True)
        # map the column index to a string
        encoded_PrimaryGenre.columns = encoded_PrimaryGenre.columns.map(lambda x: x[0])

        data=data.reset_index(level=0, drop=True)
        data=pd.concat([data, genres_df,languages_df], axis=1)

        for item in (encoded_PrimaryGenre.columns):
            if item in genres_df:
                listy = []
                for i in range(len(data)):
                    listy.append(data[item][i] + encoded_PrimaryGenre[item][i])

                data[item] = listy
            else:
                data[item] = encoded_PrimaryGenre[item].values


        target_train=terget_encoder.transform(data[target_column].applymap(lambda x: tuple(x) if isinstance(x, list) else x))
        data["Primary_Genre"]=target_train['Primary Genre'].values
        data["Developer_target"]=target_train['Developer'].values
        data["Age_Rating_target"]=target_train['Age Rating'].values
        data['Description'] = data['Description'].apply(lambda x: Cleaning_Descriptions(x))

        data['Description']=data['Description'].str.lower()
        data['Description']=data['Description'].apply(lambda x:remove_stopwords(x))
        data['Description']=data['Description'].apply(lambda x:remove_additional_space(x))
        data['Description']=data['Description'].apply(lambda x:Summarization(x))
        data['Description']=data['Description'].apply(lambda x:stemm_words(x))
        data['Description']=data['Description'].apply(lambda x:lemmatize_words(x))


        features=cv.transform(data['Description'])
        data_f_train=pd.DataFrame(features.toarray(),columns=cv.get_feature_names_out())

        data_f_train = data_f_train.reset_index(level=0, drop=True)

        data=data.reset_index(level=0, drop=True)
        data=pd.concat([data,data_f_train], axis=1)


        data=data[selected_columns]

        scaled_data=standard_scaler.transform(data)
        y=rf.predict(scaled_data)
        y=int(y)
        if y==0 :
            sl.write('Game Rate is <span style="font-size:25px; font-weight:bold">Intermediate</span>', unsafe_allow_html=True)
        elif y==1:
            sl.write('Game Rate is <span style="font-size:25px; font-weight:bold">Intermediate</span>', unsafe_allow_html=True)
        else:
            sl.write('Game Rate is <span style="font-size:25px; font-weight:bold">High</span>', unsafe_allow_html=True)
