def load_common_words():
    f = open("words.txt", "r")
    common_words = set()
    for word in f:
        common_words.add(word.strip('\n').lower())
    f.close()
    return common_words
def top_words(file_name):
    f = open(file_name, "r")
    record = {}
    common_words = load_common_words()
    for line in f:
        words = line.split()
        for word in words:
            w = word.strip('.!?,()\n').lower()
            if record.has_key(w):
                record[w] += 1
            else:
                record[w] = 1

    for word in record.keys():
        if word in common_words:
            record[word] = -1
    f.close()
    occur = [key for key in record.keys()]
    occur.sort(reverse=True, key=lambda x: record[x])
    return set(occur[: len(occur) / 10 ])

def calculate_score(sentence, metric):
    words = sentence.split()
    imp_words, total_words, begin_unimp, end, begin = [0]*5
    for word in words:
        w = word.strip('.!?,();').lower()
        end += 1
        if w in metric:
            imp_words += 1
            begin = total_words
            end = 0
        total_words += 1
    unimportant = total_words - begin - end
    if(unimportant != 0):
        return float(imp_words**2) / float(unimportant)
    return 0.0

def summarize(file_name):
    f = open(file_name, "r")
    text = ""
    for line in f:
        text += line.replace('!','.').replace('?','.').replace('\n',' ')
    f.close()
    sentences = text.split(".")
    metric = top_words(file_name)
    scores = {}
    for sentence in sentences:
        scores[sentence] = calculate_score(sentence, metric)
    top_sentences = list(sentences)                                # make a copy
    top_sentences.sort(key=lambda x: scores[x], reverse=True)      # sort by score
    top_sentences = top_sentences[:int(len(scores)*ABSTRACT_SIZE)] # get top 5%
    top_sentences.sort(key=lambda x: sentences.index(x))           # sort by occurrence
    return '. '.join(top_sentences)