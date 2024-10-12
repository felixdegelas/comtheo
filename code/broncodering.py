import numpy as np
import math

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


   # voeg hier je code toe

    return dictionary,gem_len,entropie



def genereer_canonische_Huffman(code_lengtes, alfabet):
    """
        Functie die een canonische Huffman codetabel maakt
        Input:
            code_lengtes : 1D numpy array met de lengte van het Huffmancodewoord voor elk symbool uit het alfabet (merk op dat de codewoordlengte van een symbool ook 0 kan zijn als dit symbool niet voorkomt)
            alfabet : 1D numpy array met alle mogelijke symbolen in dezelfde volgorde als code_lengtes ; in dit project is het alfabet alle getallen van 0 tot lengte alfabet-1
        Output:
            dictionary: dictionary met symbolen van het alfabet als key en het canonische codewoord als value
    """

    dictionary={}

    # voeg hier je code toe

    return dictionary


