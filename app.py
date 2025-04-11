from flask import *
import pandas as pd
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pickle import load
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

# save the model and vector
f = open("spam_model.pkl","rb")
model = load(f)
f.close()

f = open("spam_tf.pkl","rb")
tf = load(f)
f.close()

ss = SnowballStemmer("english")
sw = stopwords.words("english")

# text cleaning

def clean_message(txt):
	txt = txt.lower()
	txt = txt.replace(".","")
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = [ss.stem(t) for t in txt]
	txt = " ".join(txt)
	return txt

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
	msg = "";
	if request.method == "POST":
		txt = request.form.get("txt")
		ctxt = clean_message(txt)
		print(ctxt)
		ttxt = tf.transform([ctxt])
		print(ttxt)
		msg = model.predict(ttxt)
		print(msg)
		return render_template("index.html",msg=msg)
	else:
		return render_template("index.html")
if __name__ == "__main__":
	app.run(use_reloader=True,debug=True)

