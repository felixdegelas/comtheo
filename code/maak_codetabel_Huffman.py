import numpy as np
import math
from collections import defaultdict

def  maak_codetabel_Huffman(waarschijnlijkheden,alfabet):
    """
        Functie die een (gewone) Huffman codetabel maakt
        Input:
            waarschijnlijkheden : 1D numpy array die de waarschijnlijkheid van elk symbool bevat ( een symbool kan niet voorkomen en waarschijnlijkheid 0 hebben; deze symbolen moeten geen codewoord hebben)
            alfabet : 1D numpy array met alle mogelijke symbolen in dezelfde volgorde als rel_freq ; in dit project is het alfabet alle getallen van 0 tot lengte alfabet-1
        Output:
            dictionary: dictionary met symbolen van het alfabet als key en codewoord als value
            gem_len: gemiddelde codewoordlengte
            entropie: entropie van symbolen
    """

    dictionary = defaultdict(str)
    gem_len=0
    entropie=0

    #Uithalen van elementen met prob nul.
    null_index = np.where(waarschijnlijkheden == 0)[0]
    waarschijnlijkheden_pos = np.delete(waarschijnlijkheden,null_index)
    alfabet_pos = np.delete(alfabet,null_index)

    #bepalen van entropy
    entropie = np.sum(-1*waarschijnlijkheden_pos * np.log2(waarschijnlijkheden_pos))

    #aanmaken van dict met key: kansen, value: list van list van symbols. Symbols in één list zijn samengenomen.
    prob_dict = defaultdict(list)
    for i in range(len(waarschijnlijkheden_pos)):
        prob_dict[waarschijnlijkheden_pos[i]] += [[alfabet_pos[i]]]

    #Huffman toepassen tot er maar 1 prob overblijft / eerste prob = 1
    while (prob_dict.get(1) == None):
        lowest_prob1 = min(prob_dict.keys())
        symbols1 = prob_dict[lowest_prob1].pop()
        if (prob_dict[lowest_prob1] == []):
            del prob_dict[lowest_prob1]
        
        lowest_prob2 = min(prob_dict.keys())
        symbols2 = prob_dict[lowest_prob2].pop()
        if (prob_dict[lowest_prob2] == []):
            del prob_dict[lowest_prob2]

        prob_dict[lowest_prob1+lowest_prob2] +=[symbols1+symbols2]

        for i in symbols1:
            dictionary[i] = "0" + dictionary[i]
        
        for i in symbols2:
            dictionary[i] = "1" + dictionary[i]

    # gemiddelde woordlengte
    for i in alfabet_pos:
        gem_len += len(dictionary[i]) * waarschijnlijkheden_pos[i]

    return dictionary,gem_len,entropie



#testcode voor maak_codetabel_Huffman
#1
prob = np.array([0.1,0.1,0.1,0.2,0.5])
symbols = np.array([i for i in range(5)])

dic1,gem_len1,entropie1 = maak_codetabel_Huffman(prob,symbols)

print("probvec: ",prob)
print("symbvec: ",symbols)
print(gem_len1)
print(entropie1)
for i in dic1.keys():
    print(i,dic1[i])
print("#####################")


#2 // is dit wel juist? -> nog eens nakijken
prob = np.array([0.1,0.1,0.1,0.2,0.3,0.2])
symbols = np.array([i for i in range(6)])

dic1,gem_len1,entropie1 = maak_codetabel_Huffman(prob,symbols)

print("probvec: ",prob)
print("symbvec: ",symbols)
print(gem_len1)
print(entropie1)
for i in dic1.keys():
    print(i,dic1[i])
print("#####################")