from keybert import KeyBERT
from main import start
from main import train
model=KeyBERT(model="distilbert-base-nli-mean-tokens")
doc= """ 
   As a parent who wants to teach financial literacy to my children, I want the banking system to offer options for children's savings accounts, educational resources about budgeting and saving, parental controls for online banking, and user-friendly tools for monitoring their spending so that my children can learn about money management while using a safe and secure platform.

Acceptance Criteria:

The banking system should offer savings accounts specifically designed for children with attractive interest rates.
The educational resources should be easy to access and understand, and cover topics like budgeting, saving, and responsible credit card use.
Parental controls should allow parents to set spending limits, restrict certain transactions, and receive notifications of account activity.
The tools for monitoring spending should be user-friendly and include visualizations such as graphs or charts.
"""
file1='/Users/abine/PycharmProjects/nlp/f1.txt'
file2='/Users/abine/PycharmProjects/nlp/f2.txt'

def extract(doc,result):
    l=(model.extract_keywords(
        doc,
        top_n=10,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
    ))
    print(l)
    listToStr = ' '.join([str(elem) for elem in l])
    with open(result,'a',encoding='utf8') as outfile:
        outfile.write((listToStr)+'\n')
        outfile.flush()
    for ele in l:
        start(ele[0],result)

def out():
    doc=[]
    with open(file1,'r',encoding='utf8') as datafile:
            for row in datafile:
                if(row[0].isnumeric()) and row[0:2]!='1.':
                    str=' '.join([elem for elem in doc])
                    with open('result1.txt','a',encoding='utf8') as outfile:
                        outfile.write((str)+"\n")
                        outfile.flush()
                    print(str)
                    extract(str,'result1.txt')
                    doc=[]
                    doc.append(row)

                else:
                    doc.append(row)

    with open(file2,'r',encoding='utf8') as datafile:
            for row in datafile:
                if(row[0].isnumeric()) and row[0:2]!='1.':
                    str=' '.join([elem for elem in doc])
                    with open('result2.txt','a',encoding='utf8') as outfile:
                        outfile.write((str)+"\n")
                        outfile.flush()
                    print(str)
                    extract(str,'result2.txt')
                    doc=[]
                    doc.append(row)

                else:
                    doc.append(row)

if __name__ == '__main__':
    train()
    out()
