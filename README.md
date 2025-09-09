# Word2Vec-from-Scratch-using-CBOW
This project demonstrates how to generate **word embeddings** using a simple **feed-forward neural network** built with **Keras**. 
It uses the **continuous bag-of-words (CBOW)** approach, where context words are used to predict the center word.

---

## 📌 Project Overview

1. **Data Preprocessing**
   - Load text data (`big.txt`).
   - Convert text to lowercase and clean punctuation/special characters.
   - Tokenize sentences and words using **NLTK**.
   - Create word-to-index (`word_index`) and index-to-word (`index_word`) mappings.

2. **Feature Engineering**
   - Define a sliding window (`window_size = 2`) to extract context words.
   - Prepare training data:
     - `X_train`: Encoded context words.
     - `y_train`: Encoded target word (center word).

3. **Model Architecture**
   - Input layer: One-hot encoded vectors of context words.
   - Hidden layer: Dense layer with **ReLU** activation.
   - Output layer: Dense layer with **softmax** activation (predicts center word).
   - Loss function: **Categorical Crossentropy**.
   - Optimizer: **Adam**.

4. **Training**
   - The model is trained for `10 epochs` with a batch size of `32`.
   - Accuracy metric is tracked.

5. **Word Embeddings Extraction**
   - The learned embeddings are stored in the first weight matrix of the model.
   - Example:
     ```python
     word_embeddings = model.get_weights()[0]
     embedding_this = word_embeddings[word_index["this"]]
     ```
     <img width="677" height="358" alt="image" src="https://github.com/user-attachments/assets/2e0c5435-0156-4f7a-8894-34840e6415a2" />

---

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `tqdm`
  - `nltk`
  - `keras`
  - `tensorflow`

Install dependencies:
```python
pip install numpy pandas tqdm nltk keras tensorflow
```

Make sure to download NLTK punkt tokenizer:
```python
import nltk
nltk.download('punkt')
```

---

## 🚀 Usage

1. Place your text file (big.txt) in the project directory.
2. Run the script:
    ```bash
    python main.py
    ```
3. After training, extract embeddings for words:
    ```bash
    word_embeddings[word_index["king"]]
    ```
---

## 📊 Example Output
- Sample context & labels:
  ```python
  ['project', 'gutenberg', 'ebook', 'of'] -> 'the'
  ```
- Embedding for word "this":
  ```python
  array([-0.012, 0.034, ..., 0.056])
  ```

---

## 📂 Project Structure
  ```python
  ├── big.txt                                      # Input corpus
  ├── Word2Vec from Scratch using CBOW.ipynb       # Training script
  ├── README.md                                    # Project documentation
  ```
