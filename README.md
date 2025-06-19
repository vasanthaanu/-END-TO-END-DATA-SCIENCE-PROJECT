# -END-TO-END-DATA-SCIENCE-PROJECT
COMPANY:CODTECH IT SOLUTIONS

NAME:VASANTHA AVULURI

INTERN ID:CT04DF992

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

#DESCRIPTION

This project is developed as part of CodTech Internship Task-3, which involves building an end-to-end data science pipeline using deep learning and deploying the model using a web framework like Flask or FastAPI. The goal is to cover all stages of a real-world data science workflow: from data collection and preprocessing to model training, evaluation, and deployment as a web app/API.

🔧 Tools & Technologies Used

Python: Main programming language used for model training and backend development.

TensorFlow: Deep learning library used to build and train the CNN model.

Keras (part of TensorFlow): High-level API for building and training neural networks.

Flask: Lightweight web framework for creating the web app to serve the model.

VS Code: Integrated Development Environment (IDE) used for writing and testing the code.

HTML/CSS: For creating the basic frontend of the web application.

CIFAR-10 Dataset: Standard image classification dataset used for training the model.

Jupyter or Command Line (optional): For testing scripts during development.

💡 Project Description The main objective of this project is to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) and deploy this trained model using a web application built with Flask.

📥 Data Collection The CIFAR-10 dataset is loaded using tensorflow.keras.datasets.cifar10. It contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

⚙ Data Preprocessing Before training the model:

All image pixel values are normalized by dividing by 255.0.

The class labels are one-hot encoded using tf.keras.utils.to_categorical.

🧠 Model Training (train_cnn.py) A CNN is defined with: Three Conv2D layers (32 → 64 → 64 filters)

MaxPooling2D after the first two convolution layers

A Flatten layer followed by Dense layers for classification

The model is compiled with:

Optimizer: Adam

Loss function: Categorical crossentropy

Metric: Accuracy

The model is trained on the training data using model.fit() for 10 epochs with a validation split. After training, the model is evaluated and saved as cifar_cnn.h5.

🌐 Model Deployment (app.py) Using Flask, we built a simple web interface that allows users to: Upload an image

Pass the image to the trained model

View the predicted class and confidence score

The Flask app uses render_template() to load HTML pages, handles file uploads using request.files, and makes predictions using the loaded .h5 model.

The web app runs on localhost:5000 and is simple but effective in showcasing the model’s functionality.

🖼 Frontend UI The interface includes:

A file upload field

A "Predict" button

A result page that shows:

The predicted class (e.g., cat, airplane)

Confidence score (e.g., 0.84)

HTML templates (index.html and result.html) are used to render the UI.

✅ Outcome

✅ Built a complete deep learning classification model

✅ Deployed it using Flask

✅ Web app working locally to make predictions on uploaded images

✅ Fully meets CodTech’s requirement for an "end-to-end data science pipeline with deployment"

problems

🎓 Conclusion

This project is a complete demonstration of a machine learning model pipeline—from raw data to a fully functional web app. It reflects industry practices and serves as an impressive portfolio piece for

internships and jobs in the field of AI and Data Science.
#output
![Image](https://github.com/user-attachments/assets/d5bda407-1b24-478e-b3a8-401e13b14e1a)

![Image](https://github.com/user-attachments/assets/b67da45f-61ea-4c5d-85df-c3d73ea10408)

![Image](https://github.com/user-attachments/assets/73660aa3-e34a-4dbc-b563-b813bfbd1504)
