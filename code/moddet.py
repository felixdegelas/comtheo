import numpy as np
import math

#################################################
############## Mapping en Detectie ##############
#################################################

def mapper(bit_array,M,type):
    """
        Functie die de bits in bit_array omzet naar complexe symbolen van een bepaalde constellatie
        Input:
            bit_array = 1D numpy array met bits (0 of 1) die gemapped moeten worden
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die overeenkomen met de bits in bit_array
    """
    symbols=np.array([])

    # voeg hier je code toe

    return symbols

def demapper(symbols,M,type):
    """
        Functie die de (complexe) symbolen in symbols omzet naar de bijhorende bits
        Input:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gedemapped moeten worden
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            bit_array_demapped = 1D numpy array met bits (0 of 1) die overeenkomen met de symbolen in symbols
    """
    bit_array_demapped=np.array([])

    # voeg hier je code toe

    return bit_array_demapped

def discreet_geheugenloos_kanaal(a,sigma,A0,theta):
    """
        Functie die het discreet geheugenloos kanaal simuleert
        Input:
            a = 1D numpy array die de sequentie van datasymbolen ak bevat
            sigma = standaard deviatie van de witte ruis
            A0 = de schalingsfactor van het kanaal
            theta = de faserotatie (in radialen)
        Output:
            z = 1D numpy array die de samples van het ontvangen signaal zk bevat (aan de ontvanger)
    """
    z=np.array([])

    # voeg hier je code toe

    return z

def maak_decisie_variabele(z,A0_hat,theta_hat):
    """
        Functie die het gedecimeerde signaal z schaalt met hch_hat en de fase compenseert met rotatie over theta_hat
        Input:
            z = 1D numpy array die het gedecimeerde signaal bevat
            A0_hat = de geschatte schalingsfactor van het kanaal
            theta_hat = de geschatte faserotatie die de demodulator introduceerde (in radialen)
        Output:
            u = 1D numpy array die de decisievariabelen bevat
    """
    u=np.array([])

    # voeg hier je code toe

    return u

def decisie(u,M,type):
    """
        Functie die de decisievariabelen afbeeldt op de meest waarschijnlijke bijhorende symbolen
        Input:
            u = 1D numpy array met de decisievariabelen
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die het meest waarschijnlijk horen bij de decisievariabelen
    """
    symbols=np.array([])

    # voeg hier je code toe

    return symbols

#################################################
############## Basisbandmodulatie ###############
#################################################

def moduleerBB(a,T,Ns,alpha,Lf):
    """
        Functie die de symboolsequentie a omzet in een basisbandsignaal
        Input:
            a = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gemoduleerd moeten worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            alpha = roll-off factor van de square-root raised cosine puls
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan een zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            sBB = 1D numpy array die de samples van het gemoduleerde signaal sBB(t) bevat
    """
    sBB=np.array([])

    # voeg hier je code toe

    return sBB

def basisband_kanaal(sBB,sigma,A0,theta):
    """
        Functie die het continue-tijd basisbandkanaal simuleert
        Input:
            sBB = 1D numpy array die de samples van het gemoduleerde signaal sBB(t) bevat
            sigma = standaard deviatie van de witte ruis
            A0 = de schalingsfactor van het kanaal
            theta = de faserotatie (in radialen)
        Output:
            rBB = 1D numpy array die de samples van het ontvangen signaal rBB(t) bevat (aan de ontvanger)
    """
    rBB=np.array([])

    # voeg hier je code toe

    return rBB

def demoduleerBB(rBB,T,Ns,alpha,Lf):
    """
        Functie die het ontvangen signaal rBB(t) demoduleert
        Input:
            rBB = 1D numpy array die de samples van het ontvangen signaal rBB(t) bevat dat gedemoduleerd moet worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            alpha = roll-off factor van de square-root raised cosine puls
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan 1 zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            y = 1D numpy array die de samples van het gedemoduleerde signaal rBB(t) bevat
    """
    y=np.array([])

    # voeg hier je code toe

    return y

def decimatie(y,Ns,Lf):
    """
        Functie die de decimatie uitvoert op het gedemoduleerd signaal y(t)
        Input:
            y = 1D numpy array die de samples van het gedemoduleerde signaal y(t) bevat
            Ns = aantal samples per symboolinterval
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan 1 zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            z = 1D numpy array die de samples na decimatie bevat
    """
    z=np.array([])

    # voeg hier je code toe

    return z

#################################################
############## Draaggolfmodulatie ###############
#################################################

def moduleer(sBB,T,Ns,frequentie):
    """
        Functie die de (complexe) symbolen in symbols moduleert
        Input:
            a = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gemoduleerd moeten worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            frequentie = draaggolfrequentie
        Output:
            s = 1D numpy array die de samples van het gemoduleerde signaal s(t) bevat
    """
    s=np.array([])

    # voeg hier je code toe

    return s

def kanaal(s,sigma,A0):
    """
        Functie die het kanaal simuleert
        Input:
            s = 1D numpy array die de samples van het gemoduleerde signaal s(t) bevat
            sigma = standaard deviatie van de witte ruis
            A0 = schalingsfactor van het kanaal
        Output:
            r = 1D numpy array die de samples van het ontvangen signaal r(t) bevat (aan de ontvanger)
    """
    r=np.array([])

    # voeg hier je code toe

    return r

def demoduleer(r,T,Ns,frequentie,theta):
    """
        Functie die het ontvangen signaal r(t) demoduleert
        Input:
            r = 1D numpy array die de samples van het ontvangen signaal r(t) bevat dat gedemoduleerd moet worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            frequentie = draaggolfrequentie
            theta = onbekende faserotatie die de demodulator introduceert (in radialen)
        Output:
            rBB = 1D numpy array die de samples van het gedemoduleerde signaal r(t) bevat
    """
    rBB=np.array([])

    # voeg hier je code toe

    return rBB

def pulse(t,T,alpha):
    """ Niet aanpassen!
        Functie die de square-root raised cosine puls genereert
        Input:
            t = 1D numpy array met tijdstippen waarop de puls gesampled moet worden
            T = symboolinterval
            alpha = roll-off factor
        Output:
            p = 1D numpy array met de samples van de puls op de tijdstippen in t
    """
    een = (1-alpha)*np.sinc(t*(1-alpha)/T)
    twee = (alpha)*np.cos(math.pi*(t/T-0.25))*np.sinc(alpha*t/T-0.25)
    drie = (alpha)*np.cos(math.pi*(t/T+0.25))*np.sinc(alpha*t/T+0.25)
    p = 1/np.sqrt(T)*(een+twee+drie)
    return p








