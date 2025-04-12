# 📜 ShakespeareGPT: Attention is All You Need (Minimal GPT from Scratch)

> A PyTorch-based GPT-style language model trained on Shakespeare's text, built from scratch using the Transformer architecture.

## 🎯 Project Overview

This project is inspired by Andrej Karpathy’s [“Let’s Build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) and reimplements a minimal version of GPT using self-attention mechanisms and positional embeddings. It is trained on the Tiny Shakespeare dataset and outputs Shakespearean-style text generations.

✅ The goal is to make this a clean, **reproducible**, and **educational** research artifact that can be submitted as an arXiv-style paper.

---

## 🧠 Model Highlights

- 🔬 Pure PyTorch implementation of core Transformer blocks (no HuggingFace used)
- 🧱 Key modules: Token embeddings, positional encodings, multi-head self-attention, feedforward layers, residual connections
- 📉 Tracks training and validation loss
- 📊 Saves `loss_curve.png` showing loss over time
- 📝 Generates sample text during training (start, mid, end)
- 🔥 Supports temperature and top-k sampling for diversity control
- 📦 All configs (e.g. vocab size, embedding size, block size, number of heads/layers) are centralized in a `config` dictionary

---

## 📁 Project Structure

```
.
├── ShakespeareGPT_attentionisallyouneedimp.ipynb   # Main notebook
├── loss_curve.png                                   # Saved training/validation loss plot
├── generated_samples/                               # Optional folder to store generated texts
├── README.md
```

---

## ⚙️ Configuration

All training and model parameters are stored in a single config dictionary:

```python
config = {
    "batch_size": 64,
    "block_size": 128,
    "max_iters": 5000,
    "eval_interval": 500,
    "learning_rate": 1e-3,
    "device": 'cuda',
    "eval_iters": 200,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.2,
}
```

---

## 🚀 Sample Usage

### 📜 Text Generation
You can input any prompt and generate Shakespearean text:

```python
prompt = "KING HENRY:"
output = generate_text(prompt, temperature=0.8, top_k=40)
print(output)
```

---

## 📉 Loss Tracking

Loss is tracked and plotted using `matplotlib`:

![loss curve](loss_curve.png)

---

## 🧪 Evaluation

You can evaluate your model qualitatively by:

- Comparing text samples from early/mid/late training
- Experimenting with different sampling temperatures
- Inspecting attention weights (optional for advanced users)

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- NumPy

Install dependencies via:

```bash
pip install torch matplotlib numpy
```

---

## 🧾 Citation

If you use this project in your own work or paper, please cite:

```
@misc{shakespearegpt2025,
  author = {Your Name},
  title = {ShakespeareGPT: A Minimal Transformer from Scratch},
  year = 2025,
  url = {https://github.com/yourusername/shakespearegpt}
}
```

---

## ✍️ Author

**Your Name** – [@yourgithub](https://github.com/yourgithub)

---

## 📌 Notes

- This notebook is designed to be educational and readable.
- You can easily adapt it to train on other datasets (e.g., movie scripts, poetry).
- Great for those learning about **transformers**, **attention**, and **language modeling**.
