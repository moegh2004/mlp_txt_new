import nltk
import streamlit as st
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


# load data
@st.cache_data
def load_data() :
    path = "https://github.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/blob/master/spam.csv"
    data = pd.read_csv(path,encoding='latin-1')
    data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    data['len'] = data['v2'].apply(lambda x: len(x))
    return data


# save data
st.set_page_config(page_icon=r'https://img.icons8.com/?size=100&id=ZWIgKlS0AmPk&format=png&color=000000' , page_title = 'NLP classfication' , layout= 'wide')
data = load_data()

punctuation = string.punctuation

# fuctions to preses data
def remove_pun(text) :
    return ''.join(ch for ch in text if ch not in punctuation)

def to_lower(text):
    return text.lower()

def remove_stop_words(text):
    stop_word = stopwords.words('english')
    words = text.split()
    return " ".join([i for i in words if i not in stop_word])

# bulding the model
#======================
cv = CountVectorizer()
corpus = [i for i in data['v2']]
x = cv.fit_transform(corpus).toarray()
y = data['v1']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#=======================
clf = MultinomialNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

#=======================

menu = st.sidebar.selectbox('Select',["📊 Dataset","🤖 Train Model","✉️ Predict Message"])
if menu == '📊 Dataset' :
    st.title('Natural language processing')
    st.subheader('this websit represented by specifying whether the message ham or spam')
    st.image(r'https://liora.io/app/uploads/sites/9/2023/09/nlp.jpg')
    st.subheader('First five rows in data')
    st.write(data.head())
    st.subheader('The shape of data')
    st.write('Number of rows',data.shape[0])
    st.write('Number of columns',data.shape[1])
    

    data['v2'] = data['v2'].apply(lambda x: remove_pun(x))
    data['v2'] = data['v2'].apply(lambda x: to_lower(x))
    data['v2'] = data['v2'].apply(lambda x: remove_stop_words(x))
    
    st.subheader('Data after prossing')
    st.write(data.head())
    #stem = PorterStemmer()
    #def stem_words(text):
    #  words = text.split()
    #  return ' '.join([stem.stem(i) for i in words])
    #data['v2'] = data['v2'].apply(lambda x: stem_words(x))
    st.subheader('Distribution of SMS Messages')
    fig, ax = plt.subplots(figsize=(5, 5))  
    data["v1"].value_counts(normalize=True).plot.pie(
        autopct="%1.1f%%",
        labels=data["v1"].value_counts().index,
        startangle=90, 
        ylabel="",  
        ax=ax
        )
    ax.set_title("Distribution of SMS Messages")
    st.pyplot(fig)

elif menu == '🤖 Train Model':
    st.title('We use MultinomialNB model')
    st.subheader('Test the accuracy')

    # ===================== Accuracy Score ===================== #
    st.subheader("1) Accuracy Score")
    st.write(ac)

    # ===================== Confusion Matrix ===================== #
    st.subheader("2) Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    st.pyplot(fig)

    # ===================== Classification Report ===================== #
    st.subheader("3) Classification Report")
    st.write(cr)


elif menu == "✉️ Predict Message":

    txt = st.text_area("Enter your text")

    if st.button("Check"):
        if txt.strip() :
            
            txt = remove_pun(txt)
            txt = to_lower(txt)
            txt = remove_stop_words(txt)
    
            txt = cv.transform([txt]).toarray()
    
            pred = clf.predict(txt)
            proba = clf.predict_proba(txt)
            ham_prob = proba[0][0]
            spam_prob = proba[0][1]
            if pred[0] == "spam":
                st.error("🚨 Spam Message")
                st.subheader("Prediction Probability")
                st.write('Hame prab :',ham_prob)
                st.write('Hame prab :',spam_prob)
            else:
                st.success("✅ Ham (Not Spam)")
                st.subheader("Prediction Probability")
                st.write('Hame prab :',ham_prob)
                st.write('Spam prab :',spam_prob)
                st.balloons()
        else :
            st.error("You didn't enter any text")
