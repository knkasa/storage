#========== KNN classifier =======================
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

knn = KNeighborsTimeSeriesClassifier(n_neighbors=1)

params = {
    "distance": ['euclidean', 'dtw']
}
tuned_knn = GridSearchCV(
    knn, 
    params, 
    cv=KFold(n_splits=5)
)
tuned_knn.fit(X_train, y_train)
y_pred_knn = tuned_clf.predict(X_test)

print(tuned_knn.best_params_)

#=========== Bagging classifier =====================
from sktime.classification.ensemble import BaggingClassifier
from sktime.classification.dictionary_based import WEASEL
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)
y_train_encoded = encoder.transform(y_train)

base_clf = WEASEL(alphabet_size=3,support_probabilities=True, random_state=42)

clf = BaggingClassifier(
    base_clf, 
    n_estimators=6, # there are 6 features in total 
    n_features=1, 
    random_state=42
)
clf.fit(X_train, y_train)
y_pred_bagging = clf.predict(X_test)
y_pred_bagging = encoder.inverse_transform(y_pred_bagging)