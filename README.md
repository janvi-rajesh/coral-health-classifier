# 🪸 Deep Learning Framework for Coral Reef Health Monitoring

Multi-class coral reef health classification (**Healthy / Bleached / Dead**) using transfer learning with **ResNet-152** and **DenseNet-121**, with a Streamlit web app to try both models.

**🔗 Live demo:** [https://coral-health-classifier-6t2biplg23fafh55vmvxun.streamlit.app/]
**🤗 Trained models:** [janvi-rajesh/coral-health-models](https://huggingface.co/janvi-rajesh/coral-health-models)

> The app may be asleep if nobody has used it for a while. Click the wake-up button and wait a minute or two. The first prediction is slow while the models load.

## About

Most earlier work classifies coral only as healthy or bleached. This project adds the **Dead** class and compares two architectures under identical training conditions.

- **Dataset:** BHD Corals (Kaggle), 1,582 images: 712 Healthy, 720 Bleached, 150 Dead
- **Split:** 60% train / 10% validation / 30% test
- **Input:** 224×224 RGB, with model-specific preprocessing
- **Augmentation:** rotation (±20°), horizontal flip, zoom (±20%), shear, width/height shift
- **Model head:** Dense(256, ReLU) → BatchNorm → Dropout(0.5) → Dense(3, Softmax)
- **Training:** Adam, learning rate 0.00005, batch size 16, 30 epochs, last 30 layers fine-tuned

## Results

|    Model    | Accuracy | Precision | Recall | F1-score |
|------------ |----------|-----------|--------|----------|
| ResNet-152  | 96.42%   | 0.9652    | 0.9642 | 0.9646   |
| DenseNet-121| 95.57%   | 0.9536    | 0.9557 | 0.9535   |

The Dead class is the hardest because the dataset is imbalanced and dead coral looks similar to bleached coral.

## Run locally

```bash
git clone https://github.com/janvi-rajesh/coral-health-classifier.git
cd coral-health-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app downloads the trained models from Hugging Face the first time it runs, so you don't need to add them yourself.

## Project structure

```
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── notebooks/          # Training notebooks (ResNet-152, DenseNet-121)
└── screenshots/        # App screenshots
```

## Tech stack

Python, TensorFlow/Keras, scikit-learn, Streamlit, Hugging Face Hub

## Team

Farhana S.R., Janvi Rajesh, P. Padmakala, Ria Jose
Department of Computer Science & Engineering, Marian Engineering College, Thiruvananthapuram
