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
    entropie = -1*waarschijnlijkheden_pos * np.log2(waarschijnlijkheden_pos)

    #aanmaken van dict met key: kansen, value: list van list van symbols. Symbols in één list zijn samengenomen.
    prob_dict = defaultdict(list)
    for i in range(len(waarschijnlijkheden_pos)):
        prob_dict[waarschijnlijkheden_pos[i]] += [[alfabet_pos[i]]]

    #Huffman toepassen tot er maar 1 prob overblijft / eerste prob = 1
    while (len(prob_dict) != 1):
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

    return dictionary,gem_len,entropie

#testcode voor maak_codetabel_Huffman

prob = np.array([0.1,0.1,0.1,0.2,0.5])
symbols = np.array([i for i in range(5)])
print(prob)
print(symbols)

dic,_,ent = maak_codetabel_Huffman(prob,symbols)
print(dic)