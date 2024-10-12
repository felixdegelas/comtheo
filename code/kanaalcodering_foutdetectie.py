import numpy as np

def determine_CRC_bits(bit_array, generator):
    """
            Functie die de CRC bits (komt overeen met r(x) uit de opgave) van een bit array bepaalt
            Input:
                bit_array = 1D numpy array met bits (0 of 1) waarvoor de CRC bits moeten berekend worden; de meest linkse bit komt overeen met de coëfficient van de hoogste graad
                generator = generator polynoom van de CRC code (numpy array); de meest linkse bit komt overeen met de coëfficient van de hoogste graad
            Output:
                crc_bits = 1D numpy array met bits (0 of 1) die de crc bits bevat; de meest linkse bit komt overeen met de coëfficient van de hoogste graad
    """

    crc_bits=np.asarray([])
    # voeg je code hier toe

    return crc_bits