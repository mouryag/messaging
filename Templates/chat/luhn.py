import re
doc="""Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics."""
mat = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
k=[i if i%2==0 else i**3 for row in mat for i in row]
sentences=doc.split('.')
#print(sentences)
words=[word for sentence in sentences for word in sentence.split(' ')]
wordict={word:0 for word in words}
for word in words:
    wordict[word]+=1
for i in wordict.keys():
    wordict[i]/=len(words)
print(wordict)
stopwords = ['for', 'a', 'of', 'the', 'and', 'is','but','to', 'in', 'on', 'with','was','will','an','to','are','it']
scores=[]
for sent in sentences:
    sc=0
    for i in sent.split(' '):
        if i not in stopwords:
            sc=sc+wordict[i]
    scores.append(sc)
print(scores)
