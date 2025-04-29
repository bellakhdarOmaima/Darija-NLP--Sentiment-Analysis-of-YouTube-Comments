import re
def clean_text(text):
    # 1. Supprimer les balises HTML tout en conservant leur contenu
    text = re.sub(r'<.*?>', '', text)  # Exemple : "<a>link</a>" devient "link"

    # 2. Supprimer les caractères @ et # mais conserver le contenu du tag
    text = re.sub(r'[@#]', '', text)  # Supprime uniquement les caractères @ et #

    # 3. Supprimer les chiffres arabes et latins
    text = re.sub(r'[0-9٠-٩]', '', text)  # Supprime les chiffres 0-9 et ٠-٩

    # 4. Supprimer les caractères spéciaux, la ponctuation et les symboles
    text = re.sub(r'[^ء-ي\u0600-\u06FFa-zA-Z\s]', '', text)  # Garde les lettres arabes et latines uniquement

    # 5. Supprimer les emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticônes visage
        u"\U0001F300-\U0001F5FF"  # Symboles & pictogrammes
        u"\U0001F680-\U0001F6FF"  # Transport & cartes
        u"\U0001F1E0-\U0001F1FF"  # Drapeaux (iOS)
        u"\U00002500-\U00002BEF"  # Lignes
        u"\U00002702-\U000027B0"  # Symboles divers
        u"\U00002B50\U0001F004"  # Étoiles & carreaux
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 6. Retirer les espaces multiples et inutiles
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Fonction pour suppression des répétitions de lettres, ponctuations et espaces
def remove_duplicates(text):
    # 1. Réduire les lettres répétées (plus de 2 fois) à 2 occurrences
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Exemple : جميييل → جميل, tooop → toop

    # 2. Réduire les ponctuations répétées (comme ..., !!!!) à une seule occurrence
    text = re.sub(r'([.!?,])\1{1,}', r'\1', text)  # Exemple : !!!! → !, .... → .

    # 3. Réduire les espaces multiples en un seul espace
    text = re.sub(r'\s+', ' ', text)  # Exemple : "   " → " "

    # 4. Retirer les espaces inutiles au début et à la fin
    text = text.strip()

    return text

def simple_tokenize(text):
    # Diviser par les espaces et supprimer les caractères spéciaux
    tokens = re.findall(r'\b\w+\b', text)  # Garde uniquement les mots alphanumériques
    return tokens

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
    "عيانا", "غالبا"
]

stop_words_set = set(stop_words)

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words_set]



def normalize_tokens(tokens):
    normalized_tokens = []
    for token in tokens:
        # Normalisation des caractères arabes
        token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        token = token.replace("ة", "ه")
        token = token.replace("ى", "ي")
        token = token.replace("ؤ", "و")
        token = token.replace("ئ", "ي")

        # Suppression des diacritiques (voyelles courtes)
        token = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', token)

        normalized_tokens.append(token)
    return normalized_tokens

def preprocess_text(sentence):
    cleaned_sentence = clean_text(sentence)
    reduced_sentence = remove_duplicates(cleaned_sentence)
    tokens = simple_tokenize(reduced_sentence)
    tokens_no_stopwords = remove_stopwords(tokens)
    normalized_tokens = normalize_tokens(tokens_no_stopwords)
    return normalized_tokens