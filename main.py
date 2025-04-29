import numpy as np
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
from bert_model import model, bert_predict, label_map
from database import connect_to_database
from flask import Flask, render_template, request, redirect, url_for, flash, session
from cnn_model import cnn_predict
from lstm_cnn_model import lstm_cnn_predict
from database import store_video_trend, store_video_link
from collections import Counter
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configuration de l'API YouTubex
API_KEY = 'ur key '  # Remplacez par votre clé API YouTube
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
nltk.download('stopwords')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model_cardiff = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

stop_words = [
    "إذ", "إذا", "إذما", "إذن", "أف", "أقل", "أكثر", "ألا", "إلا", "التي", "الذي", "الذين", "إلى", "إليك",
    "إليكم", "إليكما", "إليكن", "أم", "أما", "إما", "أن", "إن", "إنا", "أنا", "أنت", "أنتم", "أنتما", "أنتن",
    "إنما", "إنه", "أنى", "آه", "آها", "أو", "أوه", "آي", "أي", "أيها", "إي", "أين", "أينما", "إيه", "بخ",
    "بس", "بعض", "بك", "بكم", "بكما", "بكن", "بل", "بلى", "بما", "بماذا", "بمن", "بنا", "به", "بها", "بهم",
    "حاشا", "حبذا", "حتى", "حيث", "حيثما", "حين", "خلا", "دون", "ذا", "ذات", "ذاك", "ذان", "ذلك", "ذين",
    "ذينك", "سوف", "سوى", "عدا", "عسى", "عل", "على", "عليك", "عليه", "عما", "عن", "عند", "غير", "فإذا",
    "فإن", "فلا", "فمن", "في", "فيم", "فيما", "فيه", "فيها", "قد", "كأن", "كأنما", "كأي", "كأين", "كذا",
    "كذلك", "كل", "كلا", "كلاهما", "كلتا", "كلما", "كليكما", "كليهما", "كم", "كما", "كي", "كيت", "كيف",
    "كيفما", "لا", "لاسيما", "لدى", "لست", "لستم", "لسنا", "لعل", "لك", "لكم", "لكما", "لكن", "لكنما",
    "لكي", "لكيلا", "لم", "لما", "لن", "هل", "هلا", "هم", "هما", "هنا", "هناك", "هنالك", "هو", "هي",
    "هيا", "هيهات", "والذي", "والذين", "وإذ", "هذا", "علاش", "قبل", "انه", "مع", "اني", "واش", "هه",
    "الي", "الا", "ما", "و", "لي", "و الله", "ليس", "نحن", "دبا", "علينا", "بالله", "ديال", "وإذا",
    "ولا", "ولو", "وما", "ومن", "وهو", "يا", "أبٌ", "أخٌ", "حمٌ", "فو", "أنتِ", "يناير", "فبراير",
    "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
    "جانفي", "فيفري", "أفريل", "جوان", "جويلية", "أوت", "دولار", "دينار", "ريال", "درهم", "ليرة",
    "جنيه", "قرش", "مليم", "فلس", "سنتيم", "يورو", "ين", "يوان", "شيكل", "واحد", "اثنان", "ثلاثة",
    "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة", "أحد", "اثنا", "إحدى", "ثلاث", "أربع",
    "خمس", "ست", "سبع", "ثماني", "تسع", "عشر", "ثمان", "سبت", "أحد", "اثنين", "ثلاثاء", "أربعاء",
    "خميس", "جمعة", "أول", "ثان", "ثاني", "ثالث", "رابع", "خامس", "سادس", "سابع", "ثامن", "تاسع",
    "عاشر", "حادي", "أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط",
    "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ء", "ى", "آ", "ؤ", "ئ", "أ", "ة",
    "نا", "ك", "كن", "ه", "فلان", "وا", "آمينَ", "آهِ", "آهٍ", "إيهٍ", "بخٍ", "وا", "ألفى", "تخذ",
    "ترك", "تعلَّم", "جعل", "حجا", "حبيب", "خال", "حسب", "درى", "رأى", "زعم", "صبر", "نَّ", "هلّا",
    "وا", "أل", "إلّا", "ت", "ك", "لمّا", "ن", "ه", "ا", "ي", "تجاه", "تلقاء", "جميع", "حسب",
    "سبحان", "شبه", "لعمر", "مثل", "أبو", "أخو", "فو", "مئة", "مئتان", "ثلاثمئة", "أربعمئة",
    "خمسمئة", "ستمئة", "سبعمئة", "ثمنمئة", "تسعمئة", "مائة", "ثلاثمائة", "أربعمائة",
    "خمسمائة", "ستمائة", "سبعمائة", "ثمانمئة", "تسعمائة", "عشرون", "ثلاثون", "اربعون",
    "خمسون", "ستون", "سبعون", "ثمانون", "تسعون", "عشرين", "ثلاثين", "اربعين", "خمسين",
    "ستين", "سبعين", "ثمانين", "تسعين", "بضع", "نيف", "أجمع", "جميع", "عامة", "عين", "نفس",
    "أصلا", "أهلا", "أيضا", "بعدا", "بغتة", "حقا", "حمدا", "خاصة", "ان", "دواليك", "باش",
    "من", "سحقا", "واللّه", "اللّه", "سرا", "سمعا", "صبرا", "صدقا", "صراحة", "طرا", "عجبا",
    "عيانا", "غالبا", "والله","انا","هاد","او","ولكن","اللي","عليها","الي","للي","ليا","بلا","يكون","اي","هي"
]
def predict_sentiment(comment):
    """
    Prédit le sentiment d'un commentaire en utilisant le modèle cardiffnlp/twitter-xlm-roberta-base-sentiment.
    Args:
        comment (str): Le commentaire à analyser.
    Returns:
        str: Le label prédit (positive, neutral, negative).
    """
    # Prétraiter le texte
    tokens = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_cardiff(**tokens)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities).item()
    # Mapper les prédictions à des labels
    labels = ["negative", "neutral", "positive"]
    return labels[predicted_class], probabilities[0].tolist()



@app.route('/')
def Accueil():
    return render_template('Accueil.html')

# Route pour accéder à home.html
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/trending_analysis')
def trending_analysis():
    db = connect_to_database()
    collection = db["videotrend"]

    # Récupérer toutes les vidéos
    videos = list(collection.find({}, {
        "video_id": 1,
        "title": 1,
        "channel_name": 1,
        "published_at": 1,
        "views": 1,
        "url": 1,
        "thumbnail_url": 1,
        "comments": 1
    }))

    def tokenize_arabic(text):
        """
        1) Garde uniquement espaces + caractères arabes + lettres latines
        2) Découpe en tokens (split)
        3) Convertit en minuscules
        4) Vérifie si c'est un stop word (si oui, on l'écarte)
        5) Normalise certains caractères arabes
        6) Supprime diacritiques
        7) Exclut les tokens trop courts (<= 1)
        """
        # a) Remplacer tout ce qui n'est pas (espace, arabe, lettres latines) par espace
        text = re.sub(r'[^ \u0600-\u06FFa-zA-Z]+', ' ', text)
        # b) Split par espace
        tokens = text.split()

        cleaned_tokens = []
        for token in tokens:
            token = token.lower()

            # 1) Vérifier stop words en premier
            if token in stop_words:
                continue

            # 2) Normalisations spécifiques
            token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            token = token.replace("ة", "ه")
            token = token.replace("ى", "ي")
            token = token.replace("ؤ", "و")
            token = token.replace("ئ", "ي")

            # 3) Supprimer diacritiques
            token = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', token)

            # 4) Exclure tokens trop courts
            if len(token) <= 1:
                continue

            cleaned_tokens.append(token)

        return cleaned_tokens

    # =================================================================
    # Word Cloud global (inchangé)
    # =================================================================
    all_comments = []
    for vid in videos:
        comments = vid.get("comments", [])
        for c in comments:
            all_comments.append(c.get("text", ""))

    # Compteur global (pour le Word Cloud GLOBAL)
    word_counter = Counter()
    for comment_text in all_comments:
        tokens = tokenize_arabic(comment_text)
        word_counter.update(tokens)

    # On récupère le top 200 mots (global)
    word_freq_list = word_counter.most_common(200)

    # =================================================================
    # Word Cloud par vidéo (AJOUT)
    # =================================================================
    for video in videos:
        # Récupérer les commentaires de cette vidéo
        video_comments = video.get("comments", [])
        tokens_this_video = []
        for c in video_comments:
            text = c.get("text", "")
            tk = tokenize_arabic(text)
            tokens_this_video.extend(tk)

        counter_vid = Counter(tokens_this_video)
        word_freq_for_this_vid = counter_vid.most_common(200)
        print("   word_freq_for_this_vid =", word_freq_for_this_vid)

        # On stocke en JSON (pour le template) :
        video["word_freq_per_video_json"] = json.dumps(word_freq_for_this_vid, ensure_ascii=False)

    # =================================================================
    #    Comptage global des sentiments (existant) + pourcentages
    # =================================================================
    global_sentiment_counts = {
        "bert": {"positive": 0, "negative": 0, "neutral": 0},
        "cnn":  {"positive": 0, "negative": 0},
        "lstm": {"positive": 0, "negative": 0}
    }
    total_comments_global = 0

    for video in videos:
        comments = video.get("comments", [])
        total_comments = len(comments)
        total_comments_global += total_comments

        # Pourcentages par vidéo (inchangé)
        video["sentiment_percentages"] = {
            "bert": {"positive": 0, "neutral": 0, "negative": 0},
            "cnn":  {"positive": 0, "negative": 0},
            "lstm": {"positive": 0, "negative": 0}
        }

        if total_comments > 0:
            sentiment_counts = {
                "bert": {"positive": 0, "neutral": 0, "negative": 0},
                "cnn":  {"positive": 0, "negative": 0},
                "lstm": {"positive": 0, "negative": 0}
            }

            for comment in comments:
                sentiment = comment.get("sentiment", {})

                # ---- Comptage global (inchangé)
                # BERT
                if "bert" in sentiment:
                    lbl = sentiment["bert"]
                    if lbl in global_sentiment_counts["bert"]:
                        global_sentiment_counts["bert"][lbl] += 1
                # CNN
                if "cnn" in sentiment:
                    lbl = sentiment["cnn"]
                    if lbl in global_sentiment_counts["cnn"]:
                        global_sentiment_counts["cnn"][lbl] += 1
                # LSTM
                if "lstm" in sentiment:
                    lbl = sentiment["lstm"]
                    if lbl in global_sentiment_counts["lstm"]:
                        global_sentiment_counts["lstm"][lbl] += 1

                # ---- Comptage par vidéo
                for model in ["bert", "cnn", "lstm"]:
                    lbl = sentiment.get(model)
                    if lbl and lbl in sentiment_counts[model]:
                        sentiment_counts[model][lbl] += 1

            # Calculer les pourcentages par vidéo
            video["sentiment_percentages"] = {
                model: {
                    sent: round((count / total_comments) * 100, 2)
                    for sent, count in sentiments.items()
                }
                for model, sentiments in sentiment_counts.items()
            }

    # =================================================================
    #  Convertir global_sentiment_counts en POURCENTAGES GLOBAUX
    # =================================================================
    cnn_total  = global_sentiment_counts["cnn"]["positive"] + global_sentiment_counts["cnn"]["negative"]
    lstm_total = global_sentiment_counts["lstm"]["positive"] + global_sentiment_counts["lstm"]["negative"]
    bert_total = (
        global_sentiment_counts["bert"]["positive"]
        + global_sentiment_counts["bert"]["negative"]
        + global_sentiment_counts["bert"]["neutral"]
    )

    global_percentages = {
        "cnn": {
            "positive": round((global_sentiment_counts["cnn"]["positive"] / cnn_total) * 100, 2) if cnn_total else 0,
            "negative": round((global_sentiment_counts["cnn"]["negative"] / cnn_total) * 100, 2) if cnn_total else 0
        },
        "lstm": {
            "positive": round((global_sentiment_counts["lstm"]["positive"] / lstm_total) * 100, 2) if lstm_total else 0,
            "negative": round((global_sentiment_counts["lstm"]["negative"] / lstm_total) * 100, 2) if lstm_total else 0
        },
        "bert": {
            "positive": round((global_sentiment_counts["bert"]["positive"] / bert_total) * 100, 2) if bert_total else 0,
            "negative": round((global_sentiment_counts["bert"]["negative"] / bert_total) * 100, 2) if bert_total else 0,
            "neutral":  round((global_sentiment_counts["bert"]["neutral"]  / bert_total) * 100, 2) if bert_total else 0
        }
    }

    print("DEBUG word_freq_list =", word_freq_list)
    print("DEBUG word_freq_for_this_vid =", word_freq_for_this_vid)

    return render_template(
        'trending_analysis.html',
        videos=videos,
        global_percentages=global_percentages,
        word_freq_list=word_freq_list
    )




@app.route('/link_analysis')
def link_analysis():
    db = connect_to_database()
    collection = db["videolink"]

    # Récupérer toutes les vidéos
    videos = list(collection.find({}, {
        "video_id": 1,
        "title": 1,
        "channel_name": 1,
        "published_at": 1,
        "views": 1,
        "url": 1,
        "scraped_at": 1,
        "thumbnail_url": 1,
        "comments": 1,
        "sentiment_summary": 1  # Ajoutez ce champ
    }))

    # Définir les stop words pour plusieurs langues
    stop_words = set()
    languages = ['english', 'french', 'arabic']
    for lang in languages:
        stop_words.update(stopwords.words(lang))

    def tokenize_multilingual(text):
        """
        1) Garde uniquement espaces + caractères alphabétiques
        2) Découpe en tokens (split)
        3) Convertit en minuscules
        4) Vérifie si c'est un stop word (si oui, on l'écarte)
        5) Normalise certains caractères arabes
        6) Supprime diacritiques (pour l'arabe)
        7) Exclut les tokens trop courts (<= 1)
        """
        # a) Remplacer tout ce qui n'est pas (espace, lettres) par espace
        text = re.sub(r'[^ \u0600-\u06FFa-zA-Z]+', ' ', text)
        # b) Split par espace
        tokens = text.split()

        cleaned_tokens = []
        for token in tokens:
            token = token.lower()

            # 1) Vérifier stop words en premier
            if token in stop_words:
                continue

            # 2) Normalisations spécifiques pour l'arabe
            token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            token = token.replace("ة", "ه")
            token = token.replace("ى", "ي")
            token = token.replace("ؤ", "و")
            token = token.replace("ئ", "ي")

            # 3) Supprimer diacritiques (pour l'arabe)
            token = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', token)

            # 4) Exclure tokens trop courts
            if len(token) <= 1:
                continue

            cleaned_tokens.append(token)

        return cleaned_tokens

    # =================================================================
    # Word Cloud global
    # =================================================================
    all_comments = []
    for vid in videos:
        comments = vid.get("comments", [])
        for c in comments:
            all_comments.append(c.get("comment_text", ""))

    # Compteur global (pour le Word Cloud GLOBAL)
    word_counter = Counter()
    for comment_text in all_comments:
        tokens = tokenize_multilingual(comment_text)
        word_counter.update(tokens)

    # On récupère le top 200 mots (global)
    word_freq_list = word_counter.most_common(200)

    # =================================================================
    # Word Cloud par vidéo (AJOUT)
    # =================================================================
    for video in videos:
        # Récupérer les commentaires de cette vidéo
        video_comments = video.get("comments", [])
        tokens_this_video = []
        for c in video_comments:
            text = c.get("comment_text", "")
            tk = tokenize_multilingual(text)
            tokens_this_video.extend(tk)

        counter_vid = Counter(tokens_this_video)
        word_freq_for_this_vid = counter_vid.most_common(200)

        # On stocke en JSON (pour le template) :
        video["word_freq_per_video_json"] = json.dumps(word_freq_for_this_vid, ensure_ascii=False)

    # =================================================================
    # Comptage global des sentiments
    # =================================================================
    global_sentiment_counts = {"positive": 0, "negative": 0}
    total_comments_global = 0

    for video in videos:
        comments = video.get("comments", [])
        total_comments = len(comments)
        total_comments_global += total_comments

        # Pourcentages par vidéo
        video["sentiment_percentages"] = {"positive": 0, "negative": 0}

        if total_comments > 0:
            sentiment_counts = {"positive": 0, "negative": 0}

            for comment in comments:
                sentiment = comment.get("sentiment", "")

                # ---- Comptage global
                if sentiment in global_sentiment_counts:
                    global_sentiment_counts[sentiment] += 1

                # ---- Comptage par vidéo
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1

            # Calculer les pourcentages par vidéo
            video["sentiment_percentages"] = {
                sent: round((count / total_comments) * 100, 2)
                for sent, count in sentiment_counts.items()
            }

    # =================================================================
    # Convertir global_sentiment_counts en POURCENTAGES GLOBAUX
    # =================================================================
    global_percentages = {
        "positive": round((global_sentiment_counts["positive"] / total_comments_global) * 100, 2) if total_comments_global else 0,
        "negative": round((global_sentiment_counts["negative"] / total_comments_global) * 100, 2) if total_comments_global else 0
    }

    return render_template(
        'link_analysis.html',
        videos=videos,
        global_percentages=global_percentages,
        word_freq_list=word_freq_list
    )

@app.route('/scraper_trending', methods=['POST'])
def scraper_trending():
    num_videos = int(request.form['num_videos'])
    periode = request.form['periode']
    channel_name = request.form.get('channel_name', '').strip().lower()
    keyword = request.form.get('keyword', '').strip().lower()
    num_comments = int(request.form.get('num_comments', 20))  # Récupérer le nombre de commentaires spécifié

    # Sauvegarder les critères dans la session
    session['num_videos'] = num_videos
    session['periode'] = periode
    session['channel_name'] = channel_name
    session['keyword'] = keyword
    session['num_comments'] = num_comments

    # Obtenir les vidéos tendances
    videos = get_youtube_trending_videos(region_code='MA', max_results=50, channel_name=channel_name, keyword=keyword)

    # Filtrer par période
    filtered_videos = []
    current_date = datetime.now()

    if periode == 'jour':
        target_date = current_date - timedelta(days=1)
    elif periode == 'semaine':
        target_date = current_date - timedelta(weeks=1)
    elif periode == 'mois':
        target_date = current_date - timedelta(days=30)
    else:
        target_date = None

    for video in videos:
        video_publish_date = datetime.strptime(video['upload_time'], '%Y-%m-%dT%H:%M:%SZ')
        if target_date and video_publish_date >= target_date:
            filtered_videos.append(video)
        if len(filtered_videos) >= num_videos:
            break

    if not filtered_videos:
        flash("Aucune vidéo trouvée selon les critères spécifiés.")
        return redirect(url_for('home'))

    # Stocker les vidéos et leurs commentaires
    for video in filtered_videos:
        video_id = video['video_id']
        comments = get_comments_for_video(video_id, num_filtered_comments=num_comments)

        # Annoter les commentaires avec les prédictions de sentiment
        labeled_comments = []
        for comment in comments:
            sentiment_bert = predict_sentiment(comment)
            sentiment_cnn = cnn_predict(comment)
            sentiment_lstm = lstm_cnn_predict(comment)
            labeled_comments.append({
                "text": comment,
                "sentiment": {
                    "bert": sentiment_bert[0],
                    "cnn": sentiment_cnn,
                    "lstm": sentiment_lstm
                }
            })

        # Stocker dans la base de données
        store_video_trend(video, labeled_comments)

    return render_template('video_trending.html', videos=filtered_videos, num_comments=num_comments)

def get_youtube_trending_videos(region_code='MA', max_results=50, channel_name=None, keyword=None):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    request = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        chart='mostPopular',
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()
    video_data = []

    for item in response['items']:
        if channel_name and item['snippet']['channelTitle'].lower() != channel_name.lower():
            continue
        if keyword and keyword.lower() not in item['snippet']['title'].lower():
            continue
        video_data.append({
            'video_id': item['id'],
            'title': item['snippet']['title'],
            'url': f"https://www.youtube.com/watch?v={item['id']}",
            'views': item['statistics']['viewCount'],
            'upload_time': item['snippet']['publishedAt'],
            'author': item['snippet']['channelTitle'],
            'thumbnail': item['snippet']['thumbnails']['default']['url']
        })

    return video_data


@app.route('/get_comments_trending/<video_id>')
def get_comments_trending(video_id):
    num_comments = int(request.args.get('num_comments', 50))

    # Récupérer les commentaires
    comments = get_comments_for_video(video_id, num_filtered_comments=num_comments)

    if comments:
        # Prédictions avec BERT
        probs_bert = bert_predict(model, comments)
        preds_bert = np.argmax(probs_bert, axis=1)
        labeled_comments_bert = [(comment, label_map[pred]) for comment, pred in zip(comments, preds_bert)]

        # Prédictions avec CNN
        cnn_preds = [cnn_predict(comment) for comment in comments]
        labeled_comments_cnn = [(comment, pred) for comment, pred in zip(comments, cnn_preds)]

        # Prédictions avec BiLSTM-CNN
        lstm_preds = [lstm_cnn_predict(comment) for comment in comments]
        labeled_comments_lstm = [(comment, pred) for comment, pred in zip(comments, lstm_preds)]

        # Calcul de la distribution des sentiments pour BERT
        total_comments = len(preds_bert)
        sentiment_counts_bert = {
            'positive': np.sum(preds_bert == 2),
            'neutral': np.sum(preds_bert == 0),
            'negative': np.sum(preds_bert == 1)
        }
        sentiment_percentages_bert = {
            'positive': round((sentiment_counts_bert['positive'] / total_comments) * 100, 2),
            'neutral': round((sentiment_counts_bert['neutral'] / total_comments) * 100, 2),
            'negative': round((sentiment_counts_bert['negative'] / total_comments) * 100, 2)
        }

        # Calcul de la distribution des sentiments pour CNN
        # Calcul de la distribution des sentiments pour CNN (Binaire : Positif, Négatif)
        sentiment_counts_cnn = {
            'positive': cnn_preds.count('positive'),
            'negative': cnn_preds.count('negative')
        }
        total_cnn = sentiment_counts_cnn['positive'] + sentiment_counts_cnn['negative']
        sentiment_percentages_cnn = {
            'positive': round((sentiment_counts_cnn['positive'] / total_cnn) * 100, 2) if total_cnn > 0 else 0,
            'negative': round((sentiment_counts_cnn['negative'] / total_cnn) * 100, 2) if total_cnn > 0 else 0
        }

        # Calcul de la distribution des sentiments pour BiLSTM-CNN (Binaire : Positif, Négatif)
        sentiment_counts_lstm = {
            'positive': lstm_preds.count('positive'),
            'negative': lstm_preds.count('negative')
        }
        total_lstm = sentiment_counts_lstm['positive'] + sentiment_counts_lstm['negative']
        sentiment_percentages_lstm = {
            'positive': round((sentiment_counts_lstm['positive'] / total_lstm) * 100, 2) if total_lstm > 0 else 0,
            'negative': round((sentiment_counts_lstm['negative'] / total_lstm) * 100, 2) if total_lstm > 0 else 0
        }

    else:
        # Si aucun commentaire n'est trouvé
        labeled_comments_bert = []
        labeled_comments_cnn = []
        labeled_comments_lstm = []
        sentiment_percentages_bert = {'positive': 0, 'neutral': 0, 'negative': 0}
        sentiment_percentages_cnn = {'positive': 0, 'neutral': 0, 'negative': 0}
        sentiment_percentages_lstm = {'positive': 0, 'neutral': 0, 'negative': 0}

    return render_template(
        'comments_video_trending.html',
        comments_bert=labeled_comments_bert,
        comments_cnn=labeled_comments_cnn,
        comments_lstm=labeled_comments_lstm,
        sentiment_percentages_bert=sentiment_percentages_bert,
        sentiment_percentages_cnn=sentiment_percentages_cnn,
        sentiment_percentages_lstm=sentiment_percentages_lstm
    )


import re
import html

def filter_comments(comments):
    """
    Filtre les commentaires en excluant ceux qui :
    - Contiennent au moins un caractère latin.
    - Contiennent uniquement des emojis.
    - Contiennent des liens ou du contenu HTML.

    Args:
        comments (list): Liste des commentaires.

    Returns:
        list: Liste des commentaires filtrés.
    """
    filtered_comments = []

    # Emoji pattern (ajout des plages manquantes, comme 💍)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticônes visage
        u"\U0001F300-\U0001F5FF"  # Symboles & pictogrammes
        u"\U0001F680-\U0001F6FF"  # Transport & cartes
        u"\U0001F1E0-\U0001F1FF"  # Drapeaux (iOS)
        u"\U00002500-\U00002BEF"  # Lignes
        u"\U00002702-\U000027B0"  # Divers
        u"\U0001F004\U00002B50"  # Carreaux et étoiles
        u"\U0001F900-\U0001F9FF"  # Divers (ex. gestes modernes)
        u"\U0001FA70-\U0001FAFF"  # Objets modernes
        u"\U0001F48D"  # 💍 (ring)
        "]+",
        flags=re.UNICODE
    )

    # Lien ou HTML pattern
    html_link_pattern = re.compile(r"<.*?>|http[s]?://\S+|www\.\S+")

    # Latin character pattern (au moins un caractère latin)
    latin_character_pattern = re.compile(r"[a-zA-Z]")

    for comment in comments:
        # Décoder les entités HTML
        decoded_comment = html.unescape(comment)

        # Vérifie si le commentaire contient uniquement des emojis
        cleaned_comment = decoded_comment.strip()
        if emoji_pattern.fullmatch(cleaned_comment) or all(emoji_pattern.match(char) for char in cleaned_comment):
            continue

        # Vérifie si le commentaire contient des liens ou du HTML
        if html_link_pattern.search(decoded_comment):
            continue

        # Vérifie si le commentaire contient au moins un caractère latin
        if latin_character_pattern.search(decoded_comment):
            continue

        # Ajouter le commentaire nettoyé
        filtered_comments.append(decoded_comment.strip())

    return filtered_comments



def get_comments_for_video(video_id, num_filtered_comments=50, max_raw_comments=200):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_raw_comments:
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_raw_comments - len(comments)),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                if len(comments) >= max_raw_comments:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Erreur lors de la récupération des commentaires pour la vidéo {video_id}: {e}")
            break

    # Appliquer le filtre sur les commentaires bruts
    filtered_comments = filter_comments(comments)

    # Assurez-vous d'avoir exactement num_filtered_comments commentaires
    if len(filtered_comments) > num_filtered_comments:
        return filtered_comments[:num_filtered_comments]
    elif len(filtered_comments) < num_filtered_comments:
        print(f"Avertissement : Seulement {len(filtered_comments)} commentaires filtrés disponibles.")
    return filtered_comments




@app.route('/trending_videos', methods=['GET'])
def trending_videos():
    # Récupérer les critères depuis la session
    num_videos = session.get('num_videos', 10)
    periode = session.get('periode', 'mois')
    channel_name = session.get('channel_name', '').strip().lower()
    keyword = session.get('keyword', '').strip().lower()
    num_comments = session.get('num_comments', 20)  # Inclure num_comments


    # Obtenir les vidéos tendances
    videos = get_youtube_trending_videos(region_code='MA', max_results=50, channel_name=channel_name, keyword=keyword)

    # Filtrer par période
    filtered_videos = []
    current_date = datetime.now()

    if periode == 'jour':
        target_date = current_date - timedelta(days=1)
    elif periode == 'semaine':
        target_date = current_date - timedelta(weeks=1)
    elif periode == 'mois':
        target_date = current_date - timedelta(days=30)
    else:
        target_date = None

    for video in videos:
        video_publish_date = datetime.strptime(video['upload_time'], '%Y-%m-%dT%H:%M:%SZ')
        if target_date and video_publish_date >= target_date:
            filtered_videos.append(video)
        if len(filtered_videos) >= num_videos:
            break

    return render_template('video_trending.html', videos=filtered_videos, num_comments=num_comments)


def filter_comments_link(comments):
    """
    Filtre les commentaires pour une vidéo YouTube en excluant ceux qui :
    - Contiennent uniquement des emojis.
    - Contiennent des liens ou du contenu HTML.

    Args:
        comments (list): Liste des commentaires.

    Returns:
        list: Liste des commentaires filtrés.
    """
    filtered_comments = []

    # Emoji pattern (ajout des plages nécessaires)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticônes visage
        u"\U0001F300-\U0001F5FF"  # Symboles & pictogrammes
        u"\U0001F680-\U0001F6FF"  # Transport & cartes
        u"\U0001F1E0-\U0001F1FF"  # Drapeaux (iOS)
        u"\U00002500-\U00002BEF"  # Lignes
        u"\U00002702-\U000027B0"  # Divers
        u"\U0001F004\U00002B50"  # Carreaux et étoiles
        u"\U0001F900-\U0001F9FF"  # Gestes modernes
        u"\U0001FA70-\U0001FAFF"  # Objets modernes
        u"\U0001F48D"  # 💍 (ring)
        "]+",
        flags=re.UNICODE
    )

    # Lien ou HTML pattern
    html_link_pattern = re.compile(r"<.*?>|http[s]?://\S+|www\.\S+")

    for comment in comments:
        decoded_comment = html.unescape(comment).strip()

        # Filtrer les commentaires contenant uniquement des emojis
        if emoji_pattern.fullmatch(decoded_comment) or all(emoji_pattern.match(char) for char in decoded_comment):
            continue

        # Filtrer les commentaires contenant des liens ou du contenu HTML
        if html_link_pattern.search(decoded_comment):
            continue

        filtered_comments.append(decoded_comment)

    return filtered_comments

def get_comments_for_videolink(video_id, num_filtered_comments=50, max_raw_comments=200):
    """
    Récupère les commentaires d'une vidéo YouTube pour un lien spécifique et applique un filtre.
    """
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_raw_comments:
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_raw_comments - len(comments)),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                if len(comments) >= max_raw_comments:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Erreur lors de la récupération des commentaires pour la vidéo {video_id}: {e}")
            break

    # Appliquer le filtre spécifique aux liens
    filtered_comments = filter_comments_link(comments)

    # Vérifiez le nombre de commentaires filtrés
    if len(filtered_comments) < num_filtered_comments:
        print(f"Avertissement : Seulement {len(filtered_comments)} commentaires filtrés disponibles.")
        flash(f"Seulement {len(filtered_comments)} commentaires disponibles après filtrage.")

    # Retourner exactement num_filtered_comments
    return filtered_comments[:num_filtered_comments]


def get_video_metadata(video_id, api_key):
    """
    Récupère les métadonnées d'une vidéo YouTube.

    Args:
        video_id (str): ID de la vidéo YouTube.
        api_key (str): Clé API YouTube Data.

    Returns:
        dict: Dictionnaire contenant les métadonnées de la vidéo ou None si non trouvée.
    """
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)

    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()

    if not response['items']:
        return None  # Vidéo non trouvée

    video_info = response['items'][0]

    metadata = {
        "title": video_info['snippet']['title'],
        "channel_name": video_info['snippet']['channelTitle'],
        "published_at": video_info['snippet']['publishedAt'],
        "views": int(video_info['statistics'].get('viewCount', 0)),
        "thumbnail_url": video_info['snippet']['thumbnails']['high']['url']
    }

    return metadata


@app.route('/scraper_link', methods=['POST'])
def scraper_link():
    video_link = request.form.get('video_link')
    num_comments = int(request.form.get('num_comments_link', 20))

    # Valider et extraire l'ID de la vidéo
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    match = youtube_regex.match(video_link)
    if not match:
        flash("Veuillez entrer un lien YouTube valide.")
        return redirect(url_for('home'))
    video_id = match.group(5)

    # Obtenir les métadonnées de la vidéo
    if not API_KEY:
        flash("Clé API YouTube non configurée.")
        return redirect(url_for('home'))

    metadata = get_video_metadata(video_id, API_KEY)
    if not metadata:
        flash("Aucune métadonnée trouvée pour cette vidéo.")
        return redirect(url_for('home'))

    # Récupérer les commentaires
    comments = get_comments_for_videolink(video_id, num_filtered_comments=num_comments)

    if not comments:
        flash("Aucun commentaire trouvé pour cette vidéo.")
        return redirect(url_for('home'))

    # Prédire les sentiments des commentaires
    predictions = [predict_sentiment(comment) for comment in comments]
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    # Formater les résultats
    labeled_comments = []
    for comment, (label, probs) in zip(comments, predictions):
        labeled_comments.append({
            "comment_text": comment,
            "sentiment": label
        })
        sentiment_counts[label] += 1

    # Calculer les pourcentages des sentiments
    total_comments = len(comments)
    sentiment_percentages = {
        "positive": round((sentiment_counts["positive"] / total_comments) * 100, 2),
        "neutral": round((sentiment_counts["neutral"] / total_comments) * 100, 2),
        "negative": round((sentiment_counts["negative"] / total_comments) * 100, 2)
    }

    # Préparer les données de la vidéo avec les métadonnées
    video_data = {
        "video_id": video_id,
        "title": metadata["title"],
        "channel_name": metadata["channel_name"],
        "published_at": metadata["published_at"],
        "views": metadata["views"],
        "thumbnail_url": metadata["thumbnail_url"],
        "url": video_link,
        "scraped_at": datetime.utcnow().isoformat()
    }

    # Appeler la fonction pour stocker les données
    store_video_link(video_data, labeled_comments)

    return render_template(
        'comments_for_videolink.html',
        comments=labeled_comments,
        video_link=video_link,
        sentiment_percentages=sentiment_percentages
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)
