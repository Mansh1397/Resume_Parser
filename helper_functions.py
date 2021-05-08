from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import io
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def read_All_CV(path):
    with open(path, 'rb') as fp:
        rsrcmgr = PDFResourceManager()
        outfp = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, outfp, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
    text = outfp.getvalue()
    return text

def preprocess_training_data_and_model_train(dir_cvs, dir_model_name):    
    lemmatizer = WordNetLemmatizer()
    dircvs = [join(dir_cvs, f) for f in listdir(dir_cvs) if isfile(join(dir_cvs, f))]
    alltext = ' '  
    for cv in dircvs:
        yd = read_All_CV(cv)
        alltext += yd + " "    
    alltext = alltext.lower()
    vector = []
    sentences = sent_tokenize(alltext)
    
    for sentence in sentences:
        temp = []
        words = word_tokenize(sentence)
        pos_tag_val = pos_tag(words)
        for word, pos_tag_val_ in zip(words,pos_tag_val):
            lemma = lemmatizer.lemmatize(word)
            if pos_tag_val_[1] == 'NN' or pos_tag_val_[1] == 'VB':
                temp.append(lemma)
        vector.append(temp)
        
    global model
    model = gensim.models.Word2Vec(vector, size=300, window=5, min_count=3, workers=4)
    model.save(dir_model_name) 

def jd_embedding_calculator(model, Jd):
    sentences = sent_tokenize(Jd)
    scores = []
    lemmatizer = WordNetLemmatizer()
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tag_val = pos_tag(words)
        for word, pos_tag_val_ in zip(words,pos_tag_val):
            lemma = lemmatizer.lemmatize(word)
            try:
                scores.append(model[lemma])
            except:
                pass
    return np.mean(scores, axis=0)

def similarity_checker(matrix):
    matchPercentage = cosine_similarity(matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2)
    return matchPercentage

def check_resume_score(dir_cvs, Jd, own_model, gensim_model=None):    
    lemmatizer = WordNetLemmatizer()
    dircvs = [join(dir_cvs, f) for f in listdir(dir_cvs) if isfile(join(dir_cvs, f))]
    alltext = ' '  
    self_model_trained_enough = False
    self_model_trained_enough_count = 0
    total_count = 0
    Scores = []
    for cv in dircvs:
        yd = read_All_CV(cv)
        alltext = yd.lower()
        resume_score = []
        sentences = sent_tokenize(alltext)
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tag_val = pos_tag(words)
            for word, pos_tag_val_ in zip(words,pos_tag_val):
                total_count += 1
                lemma = lemmatizer.lemmatize(word)
                if pos_tag_val_[1] == 'NN' or pos_tag_val_[1] == 'VB':
                    if lemma in own_model.wv.vocab:
                        resume_score.append(own_model.wv[lemma])
                        self_model_trained_enough_count += 1
                    elif lemma.lower() in own_model.wv.vocab:
                            resume_score.append(own_model.wv[lemma.lower()])
                            self_model_trained_enough_count += 1
                    elif gensim_model != None:
                        if lemma in gensim_model:
                                resume_score.append(gensim_model[lemma])
                        elif lemma.lower() in gensim_model:
                                resume_score.append(gensim_model[lemma.lower()])
                        else:
                            pass
                    else:
                            pass
                
        if(self_model_trained_enough_count == total_count):
            self_model_trained_enough = True
        else:
            pass
        
        current_resume_mean_score = np.mean(resume_score, axis=0)
                
        if self_model_trained_enough:
            job_description_embeddings = jd_embedding_calculator(own_model, Jd)
        elif gensim_model != None:
            job_description_embeddings = jd_embedding_calculator(gensim_model, Jd)
        else:
            job_description_embeddings = jd_embedding_calculator(own_model, Jd)
            
        
        current_resume_selection_score = similarity_checker([current_resume_mean_score, job_description_embeddings])
        similarity_score = "Similarity Score: " + str(current_resume_selection_score)
        Scores.append((cv, similarity_score))
        Scores.sort(key=lambda x: x[1])
        
    return Scores