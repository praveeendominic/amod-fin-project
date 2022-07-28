# import flask
from flask import Flask, url_for,redirect,render_template,request
import ktrain
import os
import tensorflow as tf
import time
import pandas as pd
import tweepy
import pandas as pd
import nltk

nltk.download("all")
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import contractions
from textblob import TextBlob

import ktrain
import os



#WSGI app
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/intro',methods=['POST','GET'])
def redirect_intro():
    # import ktrain
    # import os
    # import tensorflow as tf
    predicted_sentiment=""
    txt = str(request.form.values())
    if txt != "":
        model_path = os.path.dirname(os.path.realpath(__file__)) + '/Model_BERT'
        predictor = ktrain.load_predictor(model_path)
        model = ktrain.get_predictor(predictor.model, predictor.preproc)
        predicted_sentiment = model.predict([txt])
    return render_template('intro.html', predicted_sentiment=predicted_sentiment)

@app.route('/redirect_tmp',methods=['POST','GET'])
def redirect_tmp():
    # li = ['domu','chris']
    # import time
    # import pandas as pd
    start_time = time.time()


    # df=pd.DataFrame(data={'Name': ['Dom', 'Chris'], 'Sex': ['M', 'M']})
    df=get_tweets()
    lang_dist=pd.DataFrame({'lang': df.lang.value_counts()})

    cleaned_df = preprocess(df)
    translated_df, df_en = translate(cleaned_df)

    # if len(df.lang.unique()) > 1:
    #     translated_df=translate(cleaned_df)
    # else:
    #     translated_df=cleaned_df
    predicted_df = bert_predict(translated_df, df_en)

    exec_time=time.time() - start_time

    return render_template('tmp.html', \
                           exec_time=exec_time,\
                           tables=[df.to_html(classes='df')], titles=['Name'],\
                           tables1=[cleaned_df.to_html(classes='cleaned_df')], titles1=['Name'],\
                           tables2=[lang_dist.to_html(classes='lang_dist')], titles2=['Name'],\
                           tables3=[translated_df.to_html(classes='translated_df')], titles3=['Name'],\
                           tables4=[predicted_df.to_html(classes='predicted_df')], titles4=['Name'])

    # tables3=[translated_df.to_html(classes='translated_df')], titles3=['Name'])

    # data = pd.read_excel('dummy_data.xlsx')
    # data.set_index(['Name'], inplace=True)
    # data.index.name = None
    # females = data.loc[data.Gender == 'f']
    # males = data.loc[data.Gender == 'm']
    # return render_template('view.html', tables=[females.to_html(classes='female'), males.to_html(classes='male')],
    #                        titles=['na', 'Female surfers', 'Male surfers'])


def get_tweets():
    # import tweepy
    # import pandas as pd
    import datetime

    api_key = "dw1HOxVvtFb3YdxyWF7nh53ou"
    api_key_secret = "nVSqVaIHdXbEg63i7VP7zlW3JW0Qd0pHpj484jmOTqX10rVHkv"
    access_token = "1357032758372048896-Js0AuRIqZY0EZVl7diPS1XW8IKyaIl"
    access_token_secret = "lTDpRO6HzwgZHjA2gaXlqh7iBAfMXq3iuTgyO9BGzfYdz"

    hashtags = ['#UkraineRussiaWar', '#StopWarInUkraine', '#UkraineUnderAttack', '#StopPutinNOW', '#UkraineUnderAttack',
                '#RussianWarCrimesInUkraine']

    for i in range(len(hashtags)):
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

        # query = tweepy.Cursor(api.search, q=hashtags[i]).items(1)
        query = api.search_tweets('#UkraineRussiaWar', count=10)
        tweets1 = [{'Tweets': tweet.text, 'Timestamp': tweet.created_at, 'lang': tweet.lang, 'Location':tweet.user.location} for tweet in query]  # ,'Location':tweet.location
        # print(tweets1)

        tweet_list = []
        date_list = []
        lang_list = []
        loc_list = []
        for tweet in tweets1:
            txt = tweet.get('Tweets')
            dat = tweet.get('Timestamp').strftime('%m/%d/%Y')
            lang = tweet.get('lang')
            loc=tweet.get('Location')

            tweet_list.append(txt)
            date_list.append(dat)
            lang_list.append(lang)
            loc_list.append(loc)

        df = pd.DataFrame({'Tweet': tweet_list, 'Date': date_list, 'lang': lang_list,'location':loc_list})

    return df


def preprocess(df1):
    # import nltk
    # nltk.download("all")
    # from nltk.stem import PorterStemmer, WordNetLemmatizer
    # from nltk.corpus import stopwords
    # import re
    # import contractions
    # from textblob import TextBlob

    df = df1
    df['Tweet'] = [contractions.fix(str(x)) for x in df.Tweet.values]

    stemmer = WordNetLemmatizer()
    sentences = list(df.Tweet.values)

    for i in range(len(sentences)):
        sentences[i] = re.sub('[@]+ [a-zA-z]*|[@]+[a-zA-z]*', ' ', sentences[i])
        sentences[i] = re.sub('\s*http.*/', ' ', sentences[i], flags=re.MULTILINE)
        sentences[i] = re.sub('www.*html|www.*com', ' ', sentences[i], flags=re.MULTILINE)
        sentences[i] = re.sub('RT.*:', '', sentences[i], flags=re.MULTILINE)

        # sentences[i]=re.sub('http.*\/',' ',sentences[i], flags=re.MULTILINE)
        sentences[i] = re.sub('!|,|\.|@|!|\?|\'|\;|%|~|\*|\(|\)|#', '', sentences[i], flags=re.MULTILINE)



        words = nltk.word_tokenize(sentences[i])
        words = [stemmer.lemmatize(x) for x in words if x not in stopwords.words('english')]
        sentences[i] = ' '.join(words)
        # sentences[i]=str(TextBlob(sentences[i]).correct())
        # print(i+1)

        df['Tweet'] = sentences
    # print(df)

    return df


def translate(df1):
    # import pandas as pd
    df=df1
    df_non_en = df.loc[df.lang != 'en',]
    df_en = df.loc[df.lang == 'en',]
    lang = pd.DataFrame([['Arabic', 'ar', 'ar_AR'],
                         ['Czech', 'cs', 'cs_CZ'],
                         ['German', 'de', 'de_DE'],
                         ['English', 'en', 'en_XX'],
                         ['Spanish', 'es', 'es_XX'],
                         ['Estonian', 'et', 'et_EE'],
                         ['Finnish', 'fi', 'fi_FI'],
                         ['French', 'fr', 'fr_XX'],
                         ['Gujarati', 'gu', 'gu_IN'],
                         ['Hindi', 'hi', 'hi_IN'],
                         ['Italian', 'it', 'it_IT'],
                         ['Japanese', 'ja', 'ja_XX'],
                         ['Kazakh', 'kk', 'kk_KZ'],
                         ['Korean', 'ko', 'ko_KR'],
                         ['Lithuanian', 'lt', 'lt_LT'],
                         ['Latvian', 'lv', 'lv_LV'],
                         ['Burmese', 'my', 'my_MM'],
                         ['Nepali', 'ne', 'ne_NP'],
                         ['Dutch', 'nl', 'nl_XX'],
                         ['Romanian', 'ro', 'ro_RO'],
                         ['Russian', 'ru', 'ru_RU'],
                         ['Sinhala', 'si', 'si_LK'],
                         ['Turkish', 'tr', 'tr_TR'],
                         ['Vietnamese', 'vi', 'vi_VN'],
                         ['Chinese', 'zh', 'zh_CN'],
                         ['Afrikaans', 'af', 'af_ZA'],
                         ['Azerbaijani', 'az', 'az_AZ'],
                         ['Bengali', 'bn', 'bn_IN'],
                         ['Persian', 'fa', 'fa_IR'],
                         ['Hebrew', 'he', 'he_IL'],
                         ['Croatian', 'hr', 'hr_HR'],
                         ['Indonesian', 'id', 'id_ID'],
                         ['Georgian', 'ka', 'ka_GE'],
                         ['Khmer', 'km', 'km_KH'],
                         ['Macedonian', 'mk', 'mk_MK'],
                         ['Malayalam', 'ml', 'ml_IN'],
                         ['Mongolian', 'mn', 'mn_MN'],
                         ['Marathi', 'mr', 'mr_IN'],
                         ['Polish', 'pl', 'pl_PL'],
                         ['Pashto', 'ps', 'ps_AF'],
                         ['Portuguese', 'pt', 'pt_XX'],
                         ['Swedish', 'sv', 'sv_SE'],
                         ['Swahili', 'sw', 'sw_KE'],
                         ['Tamil', 'ta', 'ta_IN'],
                         ['Telugu', 'te', 'te_IN'],
                         ['Thai', 'th', 'th_TH'],
                         ['Tagalog', 'tl', 'tl_XX'],
                         ['Ukrainian', 'uk', 'uk_UA'],
                         ['Urdu', 'ur', 'ur_PK'],
                         ['Xhosa', 'xh', 'xh_ZA'],
                         ['Galician', 'gl', 'gl_ES'],
                         ['Slovene', 'sl', 'sl_SI']
                         ])
    lang.columns = ['lang', 'lang_code_raw', 'lang_code_bart']

    # Delete non supported languages
    l1 = list(lang.lang_code_raw)
    df_non_en = df_non_en.loc[df_non_en['lang'].isin(l1)]

    # Add bart lang codes to df
    list_lang = []
    for i in range(len(df_non_en)):
        if df_non_en.iloc[i]['lang'] in list(lang.lang_code_raw):
            list_lang.extend(list(lang.loc[lang.lang_code_raw == df_non_en.iloc[i]['lang'], 'lang_code_bart']))

    df_non_en.loc[:, 'lang_mbart'] = list_lang
    # df_non_en['lang_mbart']= list_lang

    df_tmp_non_en = df_non_en#.iloc[:5, ]

    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    list_tweet = []
    for i in range(len(df_tmp_non_en)):
        # print(i)
        tokenizer.src_lang = df_tmp_non_en.iloc[i]['lang_mbart']
        sent = df_tmp_non_en.iloc[i]['Tweet']
        # print(df_non_en.iloc[i]['lang_mbart'])
        # print(sent)
        encoded_non_eng_text = tokenizer(sent, return_tensors="pt")
        generated_tokens = model.generate(**encoded_non_eng_text,
                                          forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # print('', str(out).strip('][\''))
        list_tweet.append(str(out).strip('][\''))
        # print(list_tweet)

    df_tmp_non_en['Translated_Tweet'] = list_tweet

    return df_tmp_non_en, df_en


def bert_predict(df_tmp_non_en,df_en):
    df_non_en1 = df_tmp_non_en[['Date', 'Translated_Tweet', 'lang', 'location']]
    df_non_en1.columns = ['Date', 'Tweet', 'lang', 'location']
############
    # # import pandas as pd
    df = pd.concat([df_en, df_non_en1], axis=0)
    # import ktrain
    # import os
    import tensorflow as tf
    # model_path=os.path.dirname(os.path.realpath(__file__)) + '/Model_BERT'
    #
    #
    # predictor = ktrain.load_predictor(model_path)
    # model = ktrain.get_predictor(predictor.model, predictor.preproc)
    # predicted_sentiment = model.predict(list(df.Tweet.values))
    # df['predicted_sentiment']=predicted_sentiment


    # ##########
    from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)

    stars = classifier(list(df.Tweet.values))
    predicted_sentiment=[x.get('label') for x in stars]
    df['predicted_sentiment'] = predicted_sentiment

    df.loc[(df.predicted_sentiment == '1 star') | (df.predicted_sentiment == '2 stars'), 'predicted_sentiment'] = 'negative'
    df.loc[(df.predicted_sentiment == '4 stars') | (df.predicted_sentiment == '5 stars'), 'predicted_sentiment'] = 'positive'
    df.loc[df.predicted_sentiment == '3 stars', 'predicted_sentiment'] = 'neutral'

    return df

@app.route('/redirect_login',methods=['POST','GET'])
def redirect_login():
    return render_template('login.html')


if __name__=='__main__':
    app.run(debug=True)