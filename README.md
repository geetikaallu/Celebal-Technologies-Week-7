# Celebal-Technologies-Week-7

# Iris Flower Prediction using Streamlit

This Streamlit app allows users to input features of an Iris flower and predicts the species using a trained Random Forest model.

## Tools Used
- Python
- Scikit-learn
- Streamlit
- Seaborn, Matplotlib

## Files
- `train_model.py`: Trains and saves the model as `iris_model.pkl`
- `app.py`: The main Streamlit app
- `iris_model.pkl`: Saved trained model (generated after running `train_model.py`)

## How to Run
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
python train_model.py
streamlit run app.py
