import numpy as np
def encodeer_lineaire_blokcode(bit_array,G):
    """
            Functie die bit_array encodeert met een lineaire blokcode met generatormatrix G
            Input:
                bit_array = 1D numpy array met bits (0 of 1) met de ongecodeerde bits; merk op dat bit_array meerdere informatiewoorden kan bevatten die allemaal geconcateneerd zijn
                G = 2D numpy array met bits (0 of 1) die de generator matrix van de lineaire blokcode bevat (dimensies kxn)
            Output:
                bitenc = 1D numpy array met bits (0 of 1) die de gecodeerde bits bevat; ook hier zijn de bits van de codewoorden geconcateneerd in een 1D numpy array
    """
    bitenc=np.array([])

    # voeg hier je code toe

    return bitenc

# functie die de decoder van de uitwendige code implementeert
def decodeer_lineaire_blokcode(bit_array,H):
    """
            Functie die bit_array decodeert met een lineaire blokcode met checkmatrix H
            Input:
                bit_array = 1D numpy array met bits (0 of 1) met de gecodeerde bits; merk op dat bit_array meerdere codewoorden kan bevatten die allemaal geconcateneerd zijn
                H = 2D numpy array met bits (0 of 1) die de check matrix van de lineaire blokcode bevat (dimensies (n-k)xn)
            Output:
                bitdec = 1D numpy array met bits (0 of 1) die de ongecodeerde bits bevat; ook hier zijn de bits van de informatiewoorden geconcateneerd in een 1D numpy array
                bool_fout = 1D numpy array die voor elke informatiewoord/codewoord aangeeft of er een fout is gedetecteerd
    """
    bool_fout=True
    bitdec=np.array([])

    # voeg hier je code toe

    return bitdec,bool_fout