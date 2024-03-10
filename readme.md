# A Fake News Detection Application for English and Hindi Languages.

The Hindi Fake News Detection Framework is a project aimed at detecting fake news articles in Hindi language. It utilizes a dataset comprising nearly 4500 fact checked news articles written in Hindi. The data was scraped from  websites such as Boomlive and NDTV using Beautifulsoup and Scrapy.

For word embedding generation, the project employs IndicBERT and MuRIL libraries, specifically developed for Hindi language processing. Various classifiers including Support Vector Machines (SVM), Logistic Regression, and a simple Convolutional Neural Network (CNN) are utilized for classification purposes.

Through experimentation, the combination of MuRIL embeddings and SVM classifier has been identified as the most optimal, achieving an impressive classification accuracy of 98.58%.

To showcase the functionality of the model, a standalone Graphical User Interface (GUI) running on a local host has been developed, allowing users to interact with the system conveniently.

## Link to Paper

Here is a link to our paper: https://ieeexplore.ieee.org/document/10434587
