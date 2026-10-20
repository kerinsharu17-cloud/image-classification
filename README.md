CNN Image Classification — Localhost Dashboard

## Run in VS Code (Windows)
1. Extract the ZIP and open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Use a TensorFlow-compatible Python version. Python 3.12 is a practical choice:
   `py -3.12 -m venv .venv`
4. Activate: `.venv\\Scripts\\activate`
5. Install: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`
6. Start: `python -m streamlit run app.py`
7. Open `http://localhost:8501` if your browser does not open automatically.
8. Click **Train / Retrain CNN** in the sidebar. The first run downloads CIFAR-10 and requires internet. After training, upload a JPG/PNG image.

## If dataset download fails
Official archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Keras cache usually lives in `C:\\Users\\<your-user>\\.keras\\datasets`. Download the complete archive into that folder and retry. Keras validates/extracts it. If validation fails, download again; do not rename a ZIP file to `.tar.gz`.

## Files
- `app.py`: Streamlit dashboard with upload, prediction, model training, and graphs.
- `requirements.txt`: libraries.
- `REPORT_TEMPLATE.md`: college report template.
- `outputs/`: model and graphs are generated after training.

Note: This educational model is trained on 32×32 CIFAR-10 images. It may misclassify real-world photos; the confidence score is not a guarantee.
