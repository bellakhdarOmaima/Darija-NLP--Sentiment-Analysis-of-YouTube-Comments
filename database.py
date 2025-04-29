from pymongo import MongoClient
from datetime import datetime



from pymongo import MongoClient

def connect_to_database():
    username = "ayaennair"
    password = "aya"
    database_name = "NLP"
    uri = f"link_to_ur_db_connex"
    client = MongoClient(uri)
    return client[database_name]

def store_video_trend(video_data, comments):
    db = connect_to_database()
    collection = db["videotrend"]
    collectionlink = db["videolink"]


    # Ajouter une vérification robuste pour les clés manquantes
    video_data = {
        "video_id": video_data.get("video_id"),
        "title": video_data.get("title"),
        "channel_name": video_data.get("channel_name", "Unknown Channel"),  # Valeur par défaut
        "published_at": video_data.get("upload_time"),
        "views": video_data.get("views", 0),
        "thumbnail_url": video_data.get("thumbnail"),
        "url": video_data.get("url"),
        "scraped_at": datetime.utcnow().isoformat(),
        "comments": comments,
    }

    # Vérification des doublons et mise à jour
    existing_video = collection.find_one({"video_id": video_data["video_id"]})
    if existing_video:
        collection.replace_one({"video_id": video_data["video_id"]}, video_data)
    else:
        collection.insert_one(video_data)


def calculate_sentiment_summary(comments):
    """
    Calcule le résumé des sentiments pour une liste de commentaires.
    Args:
        comments (list): Liste des commentaires avec sentiments prédits.
    Returns:
        dict: Résumé des sentiments.
    """
    sentiment_summary = {
        "bert": {"positive": 0, "neutral": 0, "negative": 0},
        "cnn": {"positive": 0, "negative": 0},
        "lstm": {"positive": 0, "negative": 0}
    }

    for comment in comments:
        sentiment_summary['bert'][comment['sentiment']['bert']] += 1
        sentiment_summary['cnn'][comment['sentiment']['cnn']] += 1
        sentiment_summary['lstm'][comment['sentiment']['lstm']] += 1

    return sentiment_summary

def store_video_link(video_data, comments):
    """
    Stocke une vidéo et ses commentaires dans la collection videolink avec un format complet.
    Si la vidéo existe déjà, elle est mise à jour.

    Args:
        video_data (dict): Dictionnaire contenant les métadonnées de la vidéo.
        comments (list): Liste des commentaires avec sentiments.
    """
    db = connect_to_database()
    collection = db["videolink"]

    # Calculer le résumé des sentiments
    sentiment_summary = calculate_sentiment_summary_for_link(comments)

    # Préparer les données à insérer
    video_entry = {
        "video_id": video_data.get("video_id"),
        "title": video_data.get("title"),
        "channel_name": video_data.get("channel_name"),
        "published_at": video_data.get("published_at"),
        "views": video_data.get("views"),
        "thumbnail_url": video_data.get("thumbnail_url"),
        "url": video_data.get("url"),
        "scraped_at": video_data.get("scraped_at"),
        "comments": comments,
        "sentiment_summary": sentiment_summary
    }

    # Vérification des doublons et mise à jour
    existing_video = collection.find_one({"video_id": video_entry["video_id"]})
    if existing_video:
        collection.replace_one({"video_id": video_entry["video_id"]}, video_entry)
    else:
        collection.insert_one(video_entry)

def calculate_sentiment_summary_for_link(comments):
    """
    Calcule le résumé des sentiments pour une liste de commentaires.

    Args:
        comments (list): Liste des commentaires avec sentiments.

    Returns:
        dict: Pourcentages des sentiments.
    """
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for comment in comments:
        sentiment = comment.get("sentiment", "")
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    total = sum(sentiment_counts.values())
    sentiment_percentages = {
        "positive": round((sentiment_counts["positive"] / total) * 100, 2) if total else 0,
        "neutral": round((sentiment_counts["neutral"] / total) * 100, 2) if total else 0,
        "negative": round((sentiment_counts["negative"] / total) * 100, 2) if total else 0
    }
    return sentiment_percentages
