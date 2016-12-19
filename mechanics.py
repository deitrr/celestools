def norm(v):
    """
    Calculates the magnitude of a vector
    
    Parameters
    ----------
    v : np.ndarray
        Numpy array containing elements of the vector. Can be any length > 1.
    
    Returns
    -------
    np.sqrt(vsq) : float        
        Magnitude of vector
    """
    vsq = 0.0
    for k in range(len(v)):
        vsq += v[k]*v[k]
    return np.sqrt(vsq)

def xangle1(longa, argp, inc):
    """
    Component of rotation matrix used in calculating the X/Vx coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """

    return np.cos(longa) * np.cos(argp) - np.sin(longa) * np.sin(argp) * np.cos(inc)

def xangle2(longa, argp, inc):
    """
    Component of rotation matrix used in calculating the X/Vx coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """

    return -np.cos(longa) * np.sin(argp) - np.sin(longa) * np.cos(argp) * np.cos(inc)

def yangle1(longa, argp, inc):
    """
    Component of rotation matrix used in calculating the Y/Vy coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """
    return np.sin(longa) * np.cos(argp) + np.cos(longa) * np.sin(argp) * np.cos(inc)

def yangle2(longa, argp, inc):
    """
    Component of rotation matrix used in calculating the Y/Vy coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """
    return -np.sin(longa) * np.sin(argp) + np.cos(longa) * np.cos(argp) * np.cos(inc)

def zangle1(argp, inc):
    """
    Component of rotation matrix used in calculating the Z/Vz coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """
    return np.sin(argp) * np.sin(inc) 

def zangle2(argp, inc):
    """
    Component of rotation matrix used in calculating the Z/Vz coordinates of an 
    orbiting body. See eqn 2.122 in Solar System Dynamics by Murray & Dermott
    
    Parameters
    ----------
    longa : float
        Longitude of ascending node in radians
    argp : float
        Argument of pericenter in radians
    inc : float
        Inclination in radians
    """
    return np.cos(argp) * np.sin(inc)
      
def xinit(semi, ecc, eanom):
    """
    Calculates x coordinate of planet/body its own orbital plane 
    
    Parameters
    ----------
    
    """
    return semi * (np.cos(eanom) - ecc)

def yinit(semi, ecc, eanom):
    return semi * np.sqrt(1.0 - ecc * ecc) * np.sin(eanom)

def vxi(m, ms, semi, x, y, eanom):
    mu = k**2.0 * (ms + m)
    n = np.sqrt(mu / (semi)**3.0)
    return -semi * semi * n * np.sin(eanom) / np.sqrt((x*x + y*y))

def vyi(m, ms, semi, x, y, eanom, ecc):
    mu = k**2.0 * (ms + m)
    n = np.sqrt(mu / (semi)**3.0)
    return semi *semi * n * np.sqrt((1 - ecc*ecc) / ((x*x + y*y))) * np.cos(eanom)

def cross(a, b):
    x = np.zeros(3)
    x[0] = a[1] * b[2] - b[1] * a[2]
    x[1] = a[2] * b[0] - b[2] * a[0]
    x[2] = a[0] * b[1] - b[0] * a[1]
    return x

def eccanom(manom, e):
    eanom = manom
    s = 1
    cond = 1.0
    
    while np.float32(cond) != 0:
        eanom += 2.0/s * jn(s, e*s) * np.sin(s*manom)
        s += 1
        cond = jn(s, e*s)
    return eanom
    
def solve_kepler(manom, e):
    # if manom < 0.05:
#       eanom = 0.01
#     else:
    eanom = manom + np.sign(np.sin(manom))*0.85*e
    di_3 = 1.0
    
    while di_3 > 1e-15:
      fi = eanom - e*np.sin(eanom) - manom
      fi_1 = 1.0 - e*np.cos(eanom)
      fi_2 = e*np.sin(eanom)
      fi_3 = e*np.cos(eanom)
      di_1 = -fi / fi_1
      di_2 = -fi / (fi_1 + 0.5*di_1*fi_2)
      di_3 = -fi / (fi_1 + 0.5*di_2*fi_2 + 1./6.*di_2**2.0*fi_3)
      eanom = eanom + di_3
    return eanom