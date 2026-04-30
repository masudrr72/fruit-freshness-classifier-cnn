🍎 Fruit Freshness Detection (CNN)

A deep learning–based image classification project that detects whether fruits are fresh or rotten.
This model supports apples, bananas, and oranges and is deployed using Streamlit for an interactive web experience.

🚀 Features
Classifies 6 categories:
Fresh Apples
Fresh Bananas
Fresh Oranges
Rotten Apples
Rotten Bananas
Rotten Oranges
Built with Convolutional Neural Networks (CNN)
Real-time prediction via image upload
Simple and clean web interface using Streamlit

🧠 Model Details
Framework: PyTorch
Architecture: Custom CNN
Input Size: 224 × 224
Output: 6 classes

📁 Project Structure
cnn_streamlit_app/
│
├── app.py             
├── model.py           
├── utils.py            
├── model.pth           
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/fruit-freshness-classifier-cnn.git
cd fruit-freshness-classifier-cnn
2. Create environment (optional but recommended)
pip install -r requirements.txt
3. Run the app
streamlit run app.py

🖼️ How to Use:

1. Upload an image of a fruit

2. The model will analyze it

3. Get instant prediction (fresh or rotten)


📊 Dataset:

Custom dataset of fruit images
6 classes (fresh & rotten categories)

🌐 Deployment:

This app is deployed using Streamlit Cloud.
👉 (Add your live app link here)

🎯 Future Improvements:

-Add more fruit categories
-Improve model accuracy
-Mobile-friendly UI
-Add confidence score

🙌 Acknowledgements:

-PyTorch for deep learning framework
-Streamlit for easy deployment

📬 Contact:

Feel free to connect with me on LinkedIn (add your link)

⭐ If you found this project useful, consider giving it a star!
