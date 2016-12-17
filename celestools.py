# -*- coding: iso-8859-1 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from subprocess import check_output
from os import system, path
from scipy.misc import comb
from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d

k = 0.01720209895
DEG = np.pi/180.0
Gcgs = 6.67428e-8 
Gmks = 6.67428e-11
AUc = 1.49598e13
AUm = 1.49598e11
pc = 3.08568025e18
MSUNg = 1.988416e33
MSUNkg = 1.988416e30
s2d = 86400.0
#AU_k = (G*MSUN*s2d**2.0/k**2.0)**(1./3.)

def mutualinc(i1, i2, la1, la2, angUnits = 'deg'):
  if angUnits == 'deg':
    i1 *= np.pi/180. 
    i2 *= np.pi/180.
    la1 *= np.pi/180. 
    la2 *= np.pi/180.
  elif angUnits == 'rad':
    pass
  else:
    raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

  return 180.0/np.pi*np.arccos(np.cos(i1)*np.cos(i2)+\
	np.sin(i1)*np.sin(i2)*np.cos(la1-la2))

def KsemiampRV(m1, m2, a, e, inc, inUnits = 'iau', outUnits = 'mks', angUnits = 'deg'):
  #calculate RV semiamplitude in m/s
  #a should be in au, mass in solar masses
  if inUnits == 'iau':
    pass
  elif inUnits == 'mks':
    m1 /= MSUNkg
    m2 /= MSUNkg
    a /= AUm
  elif inUnits == 'cgs':
    m1 /= MSUNg
    m2 /= MSUNg
    a /= AUc
  else:
    raise ValueError('Invalid units for "inUnits". Valid options are "iau", "mks" or "cgs"')
   
  if angUnits == 'deg':
    inc *= np.pi/180. 
  elif angUnits == 'rad':
    pass
  else:
    raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

  RV = (np.sqrt(Gmks/(1-e**2))*m2*MSUNkg*np.sin(inc)*(a*AUm*(m1+m2)*MSUNkg)**(-0.5))
  if outUnits == 'mks':
    return RV
  elif outUnits == 'cgs':
    return RV*100
  else:
    raise ValueError('Invalid units for "outUnits". Valid options are "mks" or "cgs"')
    
def per2semi(ms, m, P):
    """
    Calculates the semi-major axis of a planet/body based on the period
    
    Parameters
    ----------
    ms : float
	The mass of the central object (star) in solar mass units
    m : float
	The mass of the orbiting body in solar units
    P : float
	Period of the orbiting body in days
	
    Returns
    -------
    a : float
	Semi-major axis of orbiting body in AU
    """
    a = ((P*k)**2.0*(ms+m)/(4*np.pi**2.0))**(1.0/3.0)
    return a


def semi2per(ms, m, a):
    """
    Calculates the period of a planet/body based on the semi-major axis
    
    Parameters
    ----------
    ms : float
	The mass of the central object (star) in solar mass units
    m : float
	The mass of the orbiting body in solar units
    a : float
	Semi-major axis of orbiting body in AU
    	
    Returns
    -------
    P : float
	Period of the orbiting body in days
    """
    P = 2.0*np.pi*np.sqrt(a**3.0/(k**2.0*(ms+m)))
    return P
   
   
def semi2per_mca(ms, m, a):
    """
    Calculates the period of a planet/body based on the semi-major axis
    using Barbara McArthur's G, AU and Msun
    
    Parameters
    ----------
    ms : float
	The mass of the central object (star) in solar mass units
    m : float
	The mass of the orbiting body in solar units
    a : float
	Semi-major axis of orbiting body in AU
    	
    Returns
    -------
    P : float
	Period of the orbiting body in days
    """
    P = 2.0*np.pi*np.sqrt((a*AU)**3.0/(G*(ms+m)*MSUN))/86400.0
    return P
    
    
def bjs(alpha, j, s):
    """
    Calculates Laplace coefficients based on eqn 6.68 in Solar System 
    Dynamics by Murray & Dermott
    
    Parameters
    ----------
    alpha : float
	Ratio of semi-major axes of two orbiting bodies
    j : int
	Laplace coeff parameter, in most cases related to order of expansion
    s : float
	Laplace coeff parameter, typically 1/2, 3/2, 5/2...

    Returns
    -------
    b : float
	Laplace coefficient
    """
    fac = 1.0
    sm = 1.0
    term = 1.0
    n = 1
    if j == 1:
        fac = s*alpha
    else:
        for k in range(1,j+1,1):
            fac *= (s + k - 1.0) / k * alpha
    
    while term >= 1.0e-15 * sm:
        term = 1.0
        for k in range(1, n+1, 1):
	    
            term *= (s + k - 1.0) * (s + j + k - 1.0) / (k * (j+k)) * alpha**2.0
        sm += term
        n += 1
    
    b = 2.0*fac*sm
    return b    
    
def Dnb(alpha, j, s, n):
    """
    Calculates derivatives of Laplace coefficients based on eqns 6.68 - 6.71 in 
    Solar System Dynamics by Murray & Dermott. This is a recursive function that 
    repeatedly calls bjs() or itself until n = 1. Can be used instead of bjs() by
    setting n = 0.
    
    Parameters
    ----------
    alpha : float
	Ratio of semi-major axes of two orbiting bodies
    j : int
	Laplace coeff parameter, in most cases related to order of expansion
    s : float
	Laplace coeff parameter, typically 1/2, 3/2, 5/2...
    n : int
	Order of derivative (n = 1: first derivative, n = 2: second derivative, etc.)

    Returns
    -------
    Dnb : float
	Laplace coefficient
    """
    if n < 0:
      print ("Error: n < 0 in Dnb")
    elif n == 0:
      return bjs(alpha, j, s)
    elif n == 1:
      return s * ( bjs(alpha, np.abs(j-1), s+1) - 2*alpha*bjs(alpha, j, s+1) + bjs(alpha, j+1, s+1))
    else:
      return s * ( Dnb(alpha, np.abs(j-1), s+1, n-1) - 2*alpha*Dnb(alpha, j, s+1, n-1) + Dnb(alpha, j+1, s+1, n-1) \
		- 2*(n-1)*Dnb(alpha, j, s+1, n-2) )

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
    
def eccanom_n(manom, e):
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
  
def omega(h):
    hxy = np.sqrt(h[0] * h[0] + h[1] * h[1])

    #if h[2] < 0:
     #   h[0] = -h[0]  #M&D say to switch the signs for hz < 0 (retrograde orbits)
     #   h[1] = -h[1]  #however, using this I get the asc node wrong by 180 for retrograde orbits!
		       #Also, the geometry seems to say this is wrong: 
		       #If you rotate the orbit past i = 90, Omega does NOT change positions when hz < 0

    omega = np.arctan2(h[0],-h[1])
    if omega < 0.0:
      omega += 2*np.pi
    elif omega >= 2*np.pi:
      omega -= 2*np.pi
    
    return omega

def f2eanom(f, e):
    if not isinstance(f, np.ndarray):
      f = np.array([f],dtype='float')
      
    while np.any(f < 0):
      f[f<0] += 2*np.pi

    while np.any(f > 2*np.pi):
      f[f>2*np.pi] -= 2*np.pi
      
    cosE = (np.cos(f) + e) / (1.0 + e*np.cos(f))
    eanom = np.zeros(len(f))
    
    eanom[f<np.pi] = np.arccos(cosE[f<np.pi])
    eanom[f>np.pi] = 2*np.pi - np.arccos(cosE[f>np.pi])
    return eanom

def osc2x(ms, m, a, e, inc, apc, lasn, manom, inUnits = 'iau', outUnits = 'iau', angUnits = 'deg'):
    #masses in solar units, a in AU, angles in degrees
    if not isinstance(m, np.ndarray):
      m = np.array([m])

    if not isinstance(a, np.ndarray):
      a = np.array([a])
      
    if not isinstance(e, np.ndarray):
      e = np.array([e])

    if not isinstance(inc, np.ndarray):
      inc = np.array([inc])

    if not isinstance(apc, np.ndarray):
      apc = np.array([apc])

    if not isinstance(lasn, np.ndarray):
      lasn = np.array([lasn])

    if not isinstance(manom, np.ndarray):
      manom = np.array([manom])
      
    if len(m) != len(a) or len(m) != len(e) or len(m) != len(inc) or len(m) != len(apc) or len(m) != len(lasn) or len(m) != len(manom):
      print ('not all elements have the same length!')
      return None
      
    if inUnits == 'iau':
      pass
    elif inUnits == 'mks':
      m /= MSUNkg
      a /= AUm
    elif inUnits == 'cgs':
      m /= MSUNg
      a /= AUc
    else:
      raise ValueError('Invalid units for "inUnits". Valid options are "iau", "mks", or "cgs"')

    if angUnits == 'deg':
      inc *= DEG
      apc *= DEG
      lasn *= DEG
      manom *= DEG
    elif angUnits == 'rad':
      pass
    else:
      raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

    nplanets = len(m)
    xi = np.zeros((2, nplanets))
    vi = np.zeros((2, nplanets))
    eanom = np.zeros(nplanets)
    xastro = np.zeros((3, nplanets))  #nplanets + 1 will be the star
    vastro = np.zeros((3, nplanets))
    
    #----Calc eccentric anomaly and astrocentric x,y,z coords------------
    for j in range(nplanets):
        
        eanom[j] = eccanom_n(manom[j], e[j])
                
        xi[0, j] = xinit(a[j], e[j], eanom[j])
        xi[1, j] = yinit(a[j], e[j], eanom[j])
        
        vi[0, j] = vxi(m[j], ms, a[j], xi[0,j], xi[1,j], eanom[j])
        vi[1, j] = vyi(m[j], ms, a[j], xi[0,j], xi[1,j], eanom[j], e[j])
        ang = np.zeros((3,2))
        ang[0] = [xangle1(lasn[j],apc[j],inc[j]), xangle2(lasn[j],apc[j],inc[j])]
        ang[1] = [yangle1(lasn[j],apc[j],inc[j]), yangle2(lasn[j],apc[j],inc[j])]
        ang[2] = [zangle1(apc[j], inc[j]), zangle2(apc[j], inc[j])]

        for kk in range(3):
            xastro[kk, j] = sum(ang[kk, :] * xi[:, j])
            vastro[kk, j] = sum(ang[kk, :] * vi[:, j])
            
    if outUnits == 'iau':
      pass
    elif outUnits == 'mks':
      xastro *= AUm
      vastro *= AUm/s2d
    elif outUnits == 'cgs':
      xastro *= AUc
      vastro *= AUc/s2d
    else:
      raise ValueError('Invalid units for "outUnits". Valid options are "iau", "mks", or "cgs"')

    return (xastro, vastro) #return x in AU and v in AU/day


def x2osc(ms, m, x, v, set_argp = False, argp=np.array([]), set_longa = False, longa=np.array([])):
    if not isinstance(m, np.ndarray):
      m = np.array([m])  #if m is not an array, make it one so it can be indexed
    
    if len(np.shape(x)) != 2:
      x = np.array([x]).T #if x is 1d (ie, one planet), make it 2d for indexing
    
    if len(np.shape(v)) != 2:
      v = np.array([v]).T #if v is 1d (ie, one planet), make it 2d for indexing
      
    if len(m) != np.shape(x)[1] or len(m) != np.shape(v)[1]:
      print ('m, x, and v do not have the same length!')
      return None

    a = np.zeros(len(m))
    e = np.zeros(len(m))
    inc = np.zeros(len(m))
    apc = np.zeros(len(m))
    lasn = np.zeros(len(m))
    manom = np.zeros(len(m))
    for i in range(len(m)):
        r = norm(x[:,i])
        vsq = norm(v[:,i]) * norm(v[:,i])
        rdot = sum(x[:,i] * v[:,i]) / r
        mu = k**2.0 * (ms + m[i])
        h = cross(x[:,i], v[:,i])
        hsq = norm(h) * norm(h)

        a[i] = (2.0/r - vsq/mu)**(-1.0)
        e[i] = np.sqrt(1.0 - hsq / (mu*a[i]))
        inc[i] = np.arccos(h[2] / norm(h))
        
        lasn[i] = omega(h)
        
        if np.isnan(lasn[i]):
          if set_longa == False:
            raise Exception('LongA = NaN, need to set keyword "set_longa"')
          else:
            lasn[i] = longa[i]

        sinwf = x[2,i] / (r*np.sin(inc[i]))
        coswf = (x[0,i]/r + np.sin(lasn[i]) * sinwf * np.cos(inc[i])) / np.cos(lasn[i])
        
        sinf = a[i] * (1.0 - e[i]*e[i]) * rdot / (norm(h) * e[i])
        cosf = (a[i] * (1.0 - e[i]*e[i]) / r - 1.0) / e[i]
        #print sinf, cosf
        
        
        sinw = sinwf * cosf - coswf * sinf
        cosw = sinwf * sinf + coswf * cosf
        apc[i] = np.arctan2(sinw, cosw)
        while apc[i] < 0:
            apc[i] += 2*np.pi
        while apc[i] >= 2*np.pi:
            apc[i] -= 2*np.pi
    
        if np.isnan(apc[i]):
          if set_argp == False:
            raise Exception('ArgP = NaN, need to set keyword "set_argp"')
          else:
            apc[i] = argp[i]
            
        f = np.arctan2(sinf, cosf)
        eanom = f2eanom(f, e[i])
        manom[i] = eanom - e[i] * np.sin(eanom)

        while manom[i] < 0:
            manom[i] += 2*np.pi
        while manom[i] >= 2*np.pi:
            manom[i] -= 2*np.pi
            
    if len(m) == 1:
      return (a[0], e[0], inc[0]/DEG, apc[0]/DEG, lasn[0]/DEG, manom[0]/DEG)
    else:
      return (a, e, inc/DEG, apc/DEG, lasn/DEG, manom/DEG)
      
  
def print_coords(nb_temp, nb_out, directory, m, x, v, ms = 1.0, digits = 8):
    #nb_temp should be a string, name of nbody template file
    #nb_out also a string, name of desired product
    #coords = a matrix, each row corresponding to a planet, each column a coord/mass
    if not isinstance(m, np.ndarray):
      m = np.array([m])  #if m is not an array, make it one so it can be indexed
    
    if len(np.shape(x)) != 2:
      x = np.array([x]).T #if x is 1d (ie, one planet), make it 2d for indexing
    
    if len(np.shape(v)) != 2:
      v = np.array([v]).T #if v is 1d (ie, one planet), make it 2d for indexing
      
    if len(m) != np.shape(x)[1] or len(m) != np.shape(v)[1]:
      print ('m, x, and v do not have the same length!')
      return None
      
    nplanets = len(m)
    if not path.exists(directory):
      system('mkdir ' + directory)
    #system('cp ' + nb_temp + ' ' + directory + '/'+ nb_out)
    read = open(nb_temp)
    lines = read.readlines()
    read.close()
    
    for j in range(len(lines)):
      if lines[j].split() != []:
        if lines[j].split()[0] == 'M':
          lines[j] = 'M = %#.5f\n'%ms
    
    write = open(directory+'/'+nb_out,'w')
    for j in range(len(lines)):
      write.write(lines[j])
    write.close()
    
    print2f = open(directory + '/' + nb_out, 'a')
    s = '%#.8e '+('%#.'+np.str(digits)+'f ')*6+'\n'
    
    for i in range(nplanets):
        lineout = s % (m[i], x[0,i], x[1,i], x[2,i], v[0,i], v[1,i], v[2,i])
        print2f.write(lineout)
    
    a, e, inc, argp, longa, meana = x2osc(ms, m, x, v/365.25)
    print2f.write('\n# mass           semi          ecc        inc        argp       longa       meana\n')
    if nplanets == 1:
        lineout = '# %#.8e %#.8f %#.8f %#.8f %#.8f %#.8f %#.8f\n' % \
            (m, a, e, inc, argp, longa, meana)
        print2f.write(lineout)
    else:    
      for i in range(nplanets):
        lineout = '# %#.8e %#.8f %#.8f %#.8f %#.8f %#.8f %#.8f\n' % \
            (m[i], a[i], e[i], inc[i], argp[i], longa[i], meana[i])
        print2f.write(lineout)

    print2f.close()
    
def print_oscelm(nb_temp, nb_out, directory, m, a, e, inc, argp, longa, meana):
    #nb_temp should be a string, name of nbody template file
    #nb_out also a string, name of desired product
    #coords = a matrix, each row corresponding to a planet, each column a coord/mass
    if not isinstance(m, np.ndarray):
      m = np.array([m])  #if m is not an array, make it one so it can be indexed
    if not isinstance(a, np.ndarray):
      a = np.array([a])  #if a is not an array, make it one so it can be indexed  
    if not isinstance(e, np.ndarray):
      e = np.array([e])  #if e is not an array, make it one so it can be indexed
    if not isinstance(inc, np.ndarray):
      inc = np.array([inc])  #if inc is not an array, make it one so it can be indexed
    if not isinstance(argp, np.ndarray):
      argp = np.array([argp])  #if argp is not an array, make it one so it can be indexed  
    if not isinstance(longa, np.ndarray):
      longa = np.array([longa])  #if longa is not an array, make it one so it can be indexed
    if not isinstance(meana, np.ndarray):
      meana = np.array([meana])  #if meana is not an array, make it one so it can be indexed
      
    nplanets = len(m)
    system('mkdir ' + directory)
    system('cp ' + nb_temp + ' ' + directory + '/'+ nb_out)
    print2f = open(directory + '/' + nb_out, 'a')
    
    for i in range(nplanets):
        lineout = '%#.8e %#.8f %#.8f %#.8f %#.8f %#.8f %#.8f\n' % \
            (m[i], a[i], e[i], inc[i], argp[i], longa[i], meana[i])
        print2f.write(lineout)
    
    print2f.close()
   
    
def fixangles(angles):
  if not isinstance(angles, np.ndarray):
    while angles >= 360.0:
      angles -= 360.0
    while angles < 0.0:
      angles += 360.0
      
  else:  
    while np.max(angles) >= 360.0:
      angles[np.where(angles >= 360.0)] -= 360.0
    while np.min(angles) < 0.0:
      angles[np.where(angles < 0.0)] += 360.0
  return angles
    
    
def astro2bary(ms, m, xastro, vastro):
  xcom = np.array([[np.sum(m*xastro[0,:])],[np.sum(m*xastro[1,:])],[np.sum(m*xastro[2,:])]])/(np.sum(m)+ms)
  vcom = np.array([[np.sum(m*vastro[0,:])],[np.sum(m*vastro[1,:])],[np.sum(m*vastro[2,:])]])/(np.sum(m)+ms)
  
  xbary = xastro - xcom
  vbary = vastro - vcom
    
  xs = -xcom
  vs = -vcom
  
  return (xbary, vbary, xs, vs)

def angularm_osc(ms, m, a, e, inc, apc, lasn, manom):
  xastro, vastro = osc2x(ms, m, a, e, inc, apc, lasn, manom)
  xbary, vbary, xs, vs = astro2bary(ms, m, xastro, vastro)
  
  L = ms * cross(xs, vs)
  for j in range(len(m)):
    L = L + m[j] * cross(xbary[:,j], vbary[:,j])
  
  return L
  
def angularm_mca(ms, m, a, e, inc, apc, lasn, manom):
  xastro, vastro = osc2x_mca(ms, m, a, e, inc, apc, lasn, manom)
  xbary, vbary, xs, vs = astro2bary(ms, m, xastro, vastro)
  
  L = ms * cross(xs, vs)
  
  if not isinstance(m, np.ndarray):
    m = np.array([m])
    
  for j in range(len(m)):
    L = L + m[j] * cross(xbary[:,j], vbary[:,j])
  
  return L
  
  
def angularm_x(ms, m, xastro, vastro):
  xbary, vbary, xs, vs = astro2bary(ms, m, xastro, vastro)
  
  L = ms * cross(xs, vs)
  for j in range(len(m)):
    L = L + m[j] * cross(xbary[:,j], vbary[:,j])
  
  return L

def energytot_x(ms, m, xastro, vastro):
  xbary, vbary, xs, vs = astro2bary(ms, m, xastro, vastro)
  
  E = 0.5*ms*np.dot(vs.T,vs)
  for j in range(len(m)):
    E += 0.5*m[j]*np.dot(vbary[:,j].T,vbary[:,j])
    E += -k**2*ms*m[j]/np.linalg.norm(xbary[:,j]-xs.T)
    for i in range(j+1,len(m)):
      E += -k**2*m[i]*m[j]/np.linalg.norm(xbary[:,j]-xbary[:,i])
  
  return E
  
def rotate(theta, phi, m, xv):
  if not isinstance(m, np.ndarray):
    xvtmp = np.zeros(3)
    xvtmp[0] = xv[0]*np.cos(theta) + xv[1]*np.sin(theta)
    xvtmp[1] = -xv[0]*np.sin(theta) + xv[1]*np.cos(theta)
    xvtmp[2] = xv[2]
    
    xvnew = np.zeros(3)
    xvnew[0] = xvtmp[0]*np.cos(phi)-xvtmp[2]*np.sin(phi)
    xvnew[1] = xvtmp[1]
    xvnew[2] = xvtmp[0]*np.sin(phi)+xvtmp[2]*np.cos(phi)
    
    return xvnew
    
  else:
    xvtmp = np.zeros((3,len(m)))
    xvnew = np.zeros((3,len(m)))
    for j in range(len(m)):
      xvtmp[0,j] = xv[0,j]*np.cos(theta) + xv[1,j]*np.sin(theta)
      xvtmp[1,j] = -xv[0,j]*np.sin(theta) + xv[1,j]*np.cos(theta)
      xvtmp[2,j] = xv[2,j]
        
      xvnew[0,j] = xvtmp[0,j]*np.cos(phi)-xvtmp[2,j]*np.sin(phi)
      xvnew[1,j] = xvtmp[1,j]
      xvnew[2,j] = xvtmp[0,j]*np.sin(phi)+xvtmp[2,j]*np.cos(phi)

    return xvnew

def inv_planex(ms, m, xastro, vastro):
  L = angularm_x(ms, m, xastro, vastro)
  xbary, vbary, xs, vs = astro2bary(ms, m, xastro, vastro)
  
  theta = np.arctan2(L[1],L[0])
  phi = np.arctan2(np.sqrt(L[1]**2.0+L[0]**2.0),L[2])
  
  xinv = rotate(theta, phi, m, xbary)
  vinv = rotate(theta, phi, m, vbary)
  xsi = rotate(theta, phi, ms, xs)
  vsi = rotate(theta, phi, ms, vs)
  xsi = np.array([[xsi[0]], [xsi[1]], [xsi[2]]])
  vsi = np.array([[vsi[0]], [vsi[1]], [vsi[2]]])
  
  return xinv, vinv, xsi, vsi, theta, phi


def astroinv_planex(ms, m, xastro, vastro):
  xinv, vinv, xsi, vsi, theta, phi = inv_planex(ms, m, xastro, vastro)
  xai = xinv - xsi
  vai = vinv - vsi
  
  return xai, vai
  

def angularm_sec(ms, m, a, e, inc, longa):
  hz = np.sqrt((k**2.0*(ms+m))*a*(1.0-e**2.0))  #angular m/mass of planets in own orbital planes
  hX = hz*np.sin(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hY = -hz*np.cos(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hZ = hz*np.cos(inc*np.pi/180.0)
  
  Lx = m * hX
  Ly = m * hY
  Lz = m * hZ
  
  L = np.array([np.sum(Lx), np.sum(Ly), np.sum(Lz)])
  return L
  
  
def angularm_sec_mca(ms, m, a, e, inc, argp, longa, meana):
  x, v = osc2x_mca(ms, m, a, e, inc, argp, longa, meana)
  a, e, inc, argp, longa, meana = x2osc(ms, m, x, v)
  
  hz = np.sqrt((k**2.0*(ms+m))*a*(1.0-e**2.0))  #angular m/mass of planets in own orbital planes
  hX = hz*np.sin(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hY = -hz*np.cos(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hZ = hz*np.cos(inc*np.pi/180.0)
  
  Lx = hX*m
  Ly = hY*m
  Lz = hZ*m 
  
  L = np.array([np.sum(Lx), np.sum(Ly), np.sum(Lz)])
  return L
    
    
def apply_delta(ms, m, a, e, inc, argp, longa, meana, iBody, mca = 0):
  if mca == 0:
    x, v = osc2x(ms, m, a, e, inc, argp, longa, meana)
  elif mca == 1:
    x, v = osc2x_mca(ms, m, a, e, inc, argp, longa, meana)
  
  dx = np.random.uniform(-1,1,3)
  dx *= a[iBody]/norm(dx)
  dv = np.random.uniform(-1,1,3)
  dv *= np.sqrt(k**2*(ms+m[iBody])/a[iBody])/norm(dv)
  mag = 1e-14 / norm(np.concatenate([dx,dv]))
  x[:,iBody] = x[:,iBody] + mag*dx
  v[:,iBody] = v[:,iBody] + mag*dv
  
  return x, v
  
def hill_gladman(ms, m, a, e):
  mu1 = m[0]/ms
  mu2 = m[1]/ms
  alpha = mu1+mu2
  gam1 = (1.0-e[0]**2.)**0.5
  gam2 = (1.0-e[1]**2.)**0.5
  delta = (a[1]/a[0])**0.5
  
  lhs = alpha**(-3.)*(mu1+mu2/delta**2.)*(mu1*gam1+mu2*gam2*delta)**2.0
  rhs = 1.0+3.0**(4./3)*mu1*mu2/alpha**(4./3)
  
  return lhs/rhs
  
def hill_mb(ms, m, a, e, inc, argp, longa, meana):
  xa, va = osc2x(ms,m,a,e,inc,argp,longa,meana)
  L = angularm_x(ms,m,xa,va)
  L2 = np.dot(L.T,L)

  E = energytot_x(ms,m,xa,va)
  
  Mstar = (ms*m[0]+ms*m[1]+m[0]*m[1])
  lhs = -2*(ms+m[0]+m[1])*L2*E/(k**4*Mstar**3)
  rhs = 1.0+3.0**(4./3)*m[0]*m[1]/(ms**(2./3)*(m[0]+m[1])**(4./3)) - \
        m[0]*m[1]*(11.*m[0]+7.*m[1])/(3.*ms*(m[0]+m[1])**2.)
  return lhs/rhs
  
def hill_mb2(ms, m, a, e, inc, argp, longa, meana):
  xa, va = osc2x(ms,m,a,e,inc,argp,longa,meana)
  L = angularm_x(ms,m,xa,va)
  L2 = np.dot(L.T,L)

  E = energytot_x(ms,m,xa,va)
  
  Mstar = (ms*m[0]+ms*m[1]+m[0]*m[1])
  lhs = -2*(ms+m[0]+m[1])*L2*E/(k**4*Mstar**3)
  rhs = 1.0+3.0**(4./3)*m[0]*m[1]/(ms**(2./3)*(m[0]+m[1])**(4./3)) - \
        m[0]*m[1]*(7.*m[0]+11.*m[1])/(3.*ms*(m[0]+m[1])**2.)
  
  return lhs/rhs
  
def hill_mb3(ms, m, a, e, inc, argp, longa, meana):
  xa, va = osc2x(ms,m,a,e,inc,argp,longa,meana)
  L = angularm_x(ms,m,xa,va)
  L2 = np.dot(L.T,L)

  E = energytot_x(ms,m,xa,va)
  
  Mstar = (ms*m[0]+ms*m[1]+m[0]*m[1])
  lhs = -2*(ms+m[0]+m[1])*L2*E/(k**4*Mstar**3)
  rhs = 1.0+2.0*m[0]*m[1]/(ms*(m[0]+m[1]))
  
  return lhs/rhs
  


def solve_kepler(manom, e):
# solve kepler's equation for eccentric anomaly, using mean anomaly in radians
# and eccentricity 
  eanom = manom + np.sign(np.sin(manom))*0.85*e
  #make initial guess of ecc anom = mean anom
  di_3 = 1.0
  
  while di_3 > 1e-15: #tolerance 
    fi = eanom - e*np.sin(eanom) - manom
    fi_1 = 1.0 - e*np.cos(eanom)
    fi_2 = e*np.sin(eanom)
    fi_3 = e*np.cos(eanom)
    di_1 = -fi / fi_1
    di_2 = -fi / (fi_1 + 0.5*di_1*fi_2)
    di_3 = -fi / (fi_1 + 0.5*di_2*fi_2 + 1./6.*di_2**2.0*fi_3)
    eanom = eanom + di_3
  return eanom

def ecc2true_anom(eanom, e):
# calculate true anomaly from eccentric anomaly (radians) and eccentricity
# will not work if e >= 1 !!!!!
  tanf2 = np.sqrt((1+e)/(1-e))*np.tan(eanom/2.0)
  
  f = 2.0*np.arctan(tanf2)
  
  if f < 0:
    f += 2*np.pi
    
  return f

def meanl2truea(e, argp, meanl):
# input argp and meanl in degrees
  meana = (meanl - argp)%360
  ecca = solve_kepler(meana*np.pi/180.,e)  #ecca will be in radians
  f = ecc2true_anom(ecca, e)*180./np.pi  #get true anom and convert to degrees
  
  return f #return true anomaly in degrees  
  
def radial_vel(ms,mp,P,e,argp,inc,meanl,planet=False):
#velocity in m/s of star due to perturber along line of sight
  truea = meanl2truea(e,argp,meanl)
  a = ((P*daysec)**2 * G *(ms+mp)/(4*np.pi**2))**(1./3)
  return -np.sqrt(G/((ms+mp)*(a*(1-e**2))))*mp*np.sin(inc*np.pi/180.)\
        *(np.cos((argp+truea)*np.pi/180.)+e*np.cos(argp*np.pi/180.))
        
def rv_semiamp(ms,mp,P,e,inc,planet=False):   
  a = ((P*daysec)**2 * G *(ms+mp)/(4*np.pi**2))**(1./3)
  return np.sqrt(G/(a*(ms+mp)*(1.0-e**2)))*mp*np.sin(inc*np.pi/180.)
  
  