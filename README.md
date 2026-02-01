# Sentiment Analysis NLP

This repository provides a sentiment analysis tool with an interactive web interface built using Streamlit. Easily analyze the sentiment (positive, negative, or neutral) of any text directly from your browser.

https://github.com/user-attachments/assets/b5fbbbb4-884b-41d7-a6b1-f2f297fbbcd6
## Features

- **Intuitive Streamlit interface** for easy sentiment analysis
- Quick setup instructions
- Supports custom text input
- Extendable with different NLP models

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/anass-belkhlifi/sentiment_analysis_nlp.git
cd sentiment_analysis_nlp
```

### 2. Create and activate a virtual environment (Recommended)

```bash
python -m venv venv
# On Unix or MacOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

The interface will open in your default web browser (usually at [http://localhost:8501](http://localhost:8501)).

## Example

1. Open the Streamlit app as described above.
2. Enter your text in the provided input box.
3. View the detected sentiment and the associated score.

## Customization

You can customize or improve the back-end sentiment analysis model by editing the relevant code files (for example, `app.py`).

## License

This project is licensed under the MIT License.

---

Feel free to open issues or submit pull requests!
