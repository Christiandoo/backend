import joblib, warnings
warnings.filterwarnings("ignore")
svm = joblib.load('../model/svm/svm_model.pkl')
tfidf = joblib.load('../model/svm/tfidf_vectorizer.pkl')
le = joblib.load('../model/svm/label_encoder.pkl')
print('Label classes:', list(le.classes_))
print('SVM type:', type(svm).__name__)
print('Has predict_proba:', hasattr(svm, 'predict_proba'))
if hasattr(svm, 'probability'):
    print('SVM probability param:', svm.probability)
print('TF-IDF features sample:', list(tfidf.get_feature_names_out()[:5]))
