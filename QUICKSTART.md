# 🚀 Quick Start Guide

Get up and running with the Handwritten Digit Recognizer in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train the Model

Train the CNN model on MNIST dataset (this will take a few minutes):

```bash
python train_model.py
```

**What happens:**
- Downloads MNIST dataset automatically
- Trains the CNN model
- Saves model to `models/final_model.h5`
- Shows training progress and saves plots

## Step 3: Evaluate the Model

See how well your model performs:

```bash
python evaluate_model.py
```

**Output:**
- Test accuracy and loss
- Confusion matrix
- Sample predictions
- Error analysis

## Step 4: Test Predictions

Try predicting on your own images:

```bash
python predict.py --image path/to/your/digit_image.png
```

Or test on MNIST samples:

```bash
python predict.py --mnist --num_samples 10
```

## Step 5: Launch Web App

Start the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser automatically!

## 📝 Tips

- **First time training**: The model will download MNIST (~11MB) automatically
- **Training time**: Expect 5-15 minutes depending on your hardware
- **Best results**: Use images with white background and black digits
- **Image format**: PNG, JPG, or JPEG formats are supported

## 🐛 Troubleshooting

**Model not found error:**
- Make sure you've run `train_model.py` first
- Check that `models/final_model.h5` exists

**Import errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

**Low prediction accuracy:**
- Make sure your images are clear and well-centered
- Try preprocessing your images (contrast adjustment, noise reduction)

---

**Need help?** Check the main [README.md](README.md) for detailed documentation.

