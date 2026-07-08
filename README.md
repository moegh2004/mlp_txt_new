# 📩 SMS Spam Detection using NLP

A Machine Learning web application built with **Python** and **Streamlit** that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and the **Multinomial Naive Bayes** algorithm.

---

## 🚀 Features

* 📊 Explore the SMS Spam Collection dataset.
* 🧹 Text preprocessing:

  * Remove punctuation
  * Convert text to lowercase
  * Remove English stopwords
* 🤖 Train a Multinomial Naive Bayes classifier.
* 📈 Display:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report
* ✉️ Predict whether a new SMS message is Spam or Ham.
* 📌 Show prediction probabilities using `predict_proba()`.
* 🎈 Interactive Streamlit interface.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NLTK
* Matplotlib
* Seaborn
* Joblib
* KaggleHub

---

## 📂 Project Structure

```
SMS-Spam-Detection/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── screenshots/
    ├── dataset.png
    ├── training.png
    └── prediction.png
```

---

## 📥 Dataset

SMS Spam Collection Dataset from UCI Machine Learning Repository.

The dataset is downloaded automatically using KaggleHub.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SMS-Spam-Detection.git
```

Move to the project folder:

```bash
cd SMS-Spam-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Model

Algorithm:

* Multinomial Naive Bayes

Text Vectorization:

* CountVectorizer

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 🖥️ Application Pages

### 📊 Dataset

* Display dataset
* Dataset shape
* Text preprocessing
* Pie chart showing Spam/Ham distribution

### 🤖 Train Model

* Train the classifier
* Show model accuracy
* Display confusion matrix
* Display classification report

### ✉️ Predict Message

* Enter any SMS message
* Predict Spam or Ham
* Display prediction probabilities

---

## 📸 Screenshots

Add screenshots inside the `screenshots` folder.

Example:

* Dataset Page
* Train Model Page
* Prediction Page

---

## 📌 Future Improvements

* Use TF-IDF Vectorizer.
* Compare multiple machine learning models.
* Add word cloud visualization.
* Save and load the trained model only once.
* Support Arabic text classification.
* Deploy the application online using Streamlit Community Cloud.

---

## 👨‍💻 Author

**Mohammad Bilal Gharaibeh**

Computer Science Student

GitHub:
https://github.com/moegh2004

LinkedIn:
https://www.linkedin.com/in/mohammad-gharaibah-271073258

---

## ⭐ If you like this project

Please give it a ⭐ on GitHub.
****
