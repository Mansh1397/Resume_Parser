# Resume_Parser

A basic approach for selecting resumes based on a provided job description. As the problem required checking word presence with respect to its context, Bag of Words or TF-IDF was not an option. Thus, I have tried the Word2Vec model in this code. The code uses two different Word2Vec models, one is a pre-trained Gensim Word2Vec model based on Google News Vectors, while the other one is a self-trained model trained on 5 data science resumes. The code checks if Gensim's model is present or not, and if absent, resorts to using a self-trained model. Even if it is present, the first priority is given to the self-trained model and in case there is some word in the test resume that is not present in the self-trained model's vocabulary, then it switches to using Gensim's model. In case of the job description, if all the words in the test resume are present in the self-trained model, then the job description word embedding is calculated using the self-trained model, else, the gensim model is used here. Finally, the average (mean) value for job descriptions embedding is compared against the average (mean) value of test resume embedding using cosine similarity. Based on this, the resumes are ranked.


In order to download the gensim pre-trained model, please follow the link: 
https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

Note: The code can work without this model as well. In order to change the job description, please place the new one in the "Job Description.docx" file in the folder. Also, please
make sure to re-train the model in case you add/change the training resumes and place the test resumes in the "Test_Resume" folder.

