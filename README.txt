Folder Structure
AI_Assignment/
├── shared/
│   ├── AI_dataset.csv          # dataset (must be present)
│   ├── preprocessing.py        # run first — generates shared pkl files
│   ├── postprocessing.py       # shared logic used by all 3 models
│   └── ...                     # generated: X_train.pkl, tfidf_vectorizer.pkl, etc.
├── naive_bayes/
│   ├── nb_model.py             # training + evaluation
│   └── nb_infer.py             # single-message prediction (used by UI)
├── svm/
│   ├── svm_model.py
│   └── svm_infer.py
├── lstm/
│   ├── lstm_model.py
│   └── lstm_infer.py
├── model_comparison/
│   └── model_comparison.py     # cross-model comparison, run last
├── streamlit/                  # adjust folder name below if different
│   └── streamlit_app.py        # Streamlit UI
├── requirements.txt
└── README.md
Requirement 
Install all dependencies with pip install -r requirement.txt