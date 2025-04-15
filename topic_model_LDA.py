from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Use either janome or spacy with ginza for Japanese tokenizer.

# Sample Japanese documents
documents = [
    "機械学習は人工知能の一分野です。",
    "ディープラーニングはニューラルネットワークを使います。",
    "自然言語処理はAIの重要な応用です。",
    "データサイエンスの本が人気です。",
    "機械学習の応用はさまざまな分野に広がっています。",
    "ニューラルネットワークはパターン認識に強いです。"
]

# Step 1: Tokenizer for Japanese using Janome
janome_tokenizer = Tokenizer()

def tokenize(text):
    tokens = [token.base_form for token in janome_tokenizer.tokenize(text)
              if token.base_form != "*" and token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    return tokens

# Step 2: Custom tokenizer for CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(documents)

# Step 3: LDA model
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Step 4: Show topics
def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nトピック {topic_idx + 1}:")
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(" ".join(top_features))

n_top_words = 5
feature_names = vectorizer.get_feature_names_out()
print_topics(lda, feature_names, n_top_words)
