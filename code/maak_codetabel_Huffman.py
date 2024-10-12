import numpy as np
import math

rij1 = np.array([1,2])

print(rij1+np.log2(rij1))

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





    dictionary={}
    gem_len=0
    entropie=0

    #Uithalen van elementen met prob nul.
    null_index = np.where(rij1 == 0)[0]
    waarschijnlijkheden_pos = np.delete(waarschijnlijkheden_pos,null_index)
    alfabet_pos = np.delete(alfabet,null_index)

    #bepalen van entropy
    entropie = -1*waarschijnlijkheden_pos * np.log2(waarschijnlijkheden_pos)

    

    return dictionary,gem_len,entropie