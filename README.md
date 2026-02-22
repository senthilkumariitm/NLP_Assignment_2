# NLP_Assignment_2
Assignment 2 on NLP
Name: Senthil Kumar
Course: Natural Language Processing
Program: MS in Data Science & AI
University: University of Central Missouri
Semester: Spring 2026

# Bigram Language Model using Maximum Likelihood Estimation (MLE)
# Student: Senthil Kumar
# Course: Natural Language Processing
# University: University of Central Missouri

from collections import defaultdict


# Step 1: Define Training Corpus

corpus = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>",
    "<s> deep learning is fun </s>"
]

# Dictionaries to store counts
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

# Step 2: Compute Unigram and Bigram Counts

for sentence in corpus:
    words = sentence.split()
    
    # Count each word (unigram)
    for word in words:
        unigram_counts[word] += 1
    
    # Count word pairs (bigram)
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_counts[bigram] += 1


# Step 3: Bigram Probability using MLE
# P(w2 | w1) = Count(w1, w2) / Count(w1)

def bigram_probability(w1, w2):
    if unigram_counts[w1] == 0:
        return 0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


# Step 4: Sentence Probability Function
# Multiply all bigram probabilities

def sentence_probability(sentence):
    words = sentence.split()
    probability = 1.0
    
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i+1]
        probability *= bigram_probability(w1, w2)
    
    return probability


# Step 5: Test Sentences

sentence1 = "<s> I love NLP </s>"
sentence2 = "<s> I love deep learning </s>"

prob1 = sentence_probability(sentence1)
prob2 = sentence_probability(sentence2)


# Step 6: Print Results

print("Sentence 1 Probability:", prob1)
print("Sentence 2 Probability:", prob2)

if prob1 > prob2:
    print("The model prefers Sentence 1 because it has higher probability.")
else:
    print("The model prefers Sentence 2 because it has higher probability.")
