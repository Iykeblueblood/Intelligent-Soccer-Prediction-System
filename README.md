# ⚽ Intelligent Soccer Prediction System

An advanced football match prediction system that leverages a multi-layered analytical approach. This application combines a sophisticated point-based rule engine with a custom-trained machine learning model to generate quantitative predictions, which are then synthesized and given a final qualitative verdict by Google's Gemini API.


### 🚀 Live

**[View the live application here](https://intelligent-soccer-prediction-system.streamlit.app/))** 
*(Remember to replace this link with your actual Streamlit Community Cloud URL after you deploy it!)*


### ✨ Key Features

*   **Dual Prediction Engines:** The system doesn't rely on a single method. It uses:
    1.  **A Point-Based Rule Engine:** A meticulously crafted set of over 30 rules that analyze team form, offensive/defensive matchups, and other factors to create a transparent, reason-based score.
    2.  **A Machine Learning Model:** A `RandomForestClassifier` trained on years of historical league data to identify complex statistical patterns.
*   **AI Expert Verdict:** Utilizes the **Google Gemini API** to provide a final, human-like analysis that considers the predictions of both engines, detailed recent form (scorelines), and historical head-to-head context.
*   **Multi-League Support:** Fully functional for the English Premier League, Spanish La Liga, and German Bundesliga.
*   **Historical Head-to-Head Analysis:** Leverages a comprehensive historical dataset to analyze past meetings between teams, providing crucial context for the AI verdict.
*   **Interactive Web Interface:** Built with Streamlit for a clean, user-friendly, and responsive experience.
