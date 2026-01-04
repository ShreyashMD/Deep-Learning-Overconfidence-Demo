# Deep Learning Overconfidence Demo 🤖

A Streamlit application demonstrating how **uncalibrated Deep Learning models** can be **extremely confident even when they are wrong**. This project uses a simple CNN trained on the CIFAR-10 dataset to visualize the phenomenon of Softmax Overconfidence.

## 🎯 Key Concept

Standard Neural Networks trained with Cross-Entropy Loss tend to be overconfident. They push probabilities towards 1 (correct class) and 0 (others) to minimize loss, often resulting in >90% confidence scores even for incorrect predictions on out-of-distribution or ambiguous data.

## ✨ Features

- **Interactive Inference**: Upload your own images or test with random samples from the CIFAR-10 test set.
- **Real-time Visualization**: See the probability distribution for the top-5 classes.
- **Temperature Scaling**: Adjust the Softmax temperature on the fly to see how it affects confidence distribution (without changing the model's rank order).
- **Overconfidence Warning**: Automatically detects and warns when the model is highly confident (>90%) but predicts the wrong class.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd Deep-Learning
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📂 Project Structure

- `app.py`: Main Streamlit application entry point.
- `model.py`: PyTorch definition of the SimpleCNN model.
- `train.py`: Script used to train the model on CIFAR-10.
- `cifar10_model.pth`: Pre-trained model weights.
- `requirements.txt`: Python package dependencies.
- `download_data.py`: Helper script to download/verify the dataset.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
