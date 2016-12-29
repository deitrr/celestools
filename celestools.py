# -*- coding: iso-8859-1 -*-
import numpy as np
import mechanics as mech
from copy import deepcopy

k = 0.01720209895
DEG = np.pi/180.0
G = 6.67428e-11
AU = 1.49598e11
pc = 3.08568025e18
MSUN = 1.988416e30
MEARTH = 5.972186e24
s2d = 86400.0

def MutualInc(i1, i2, la1, la2, angUnits = 'deg'):
  """
  Calculates the mutual inclination of two orbiters
  
  Parameters
  ----------
  i1 : float
      Inclination of the first body
  i2 : float
      Inclination of the second body
  la1 : float
      Longitude of Ascending Node of the first body
  la2 : float
      Longitude of Ascending Node of the second body
      
  Keyword Arguments
  -----------------
  angUnits : string
      Units for angles. Options are 'deg' (default) or 'rad'
      
  Returns
  -------
  Phi : float
      Mutual (relative) inclination
  """
  if angUnits == 'deg':
    i1 *= DEG
    i2 *= DEG
    la1 *= DEG
    la2 *= DEG
  elif angUnits == 'rad':
    pass
  else:
    raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

  return 180.0/np.pi*np.arccos(np.cos(i1)*np.cos(i2)+\
        np.sin(i1)*np.sin(i2)*np.cos(la1-la2))
        
        
def RV_SemiAmp(m1, m2, a, e, inc, inUnits = 'iau', outUnits = 'mks', angUnits = 'deg'):
  """
  Calculates the radial velocity semi-amplitude of a star with a companion
  
  Parameters
  ----------
  m1 : float
      The mass of the central object (star)
  m2 : float
      The mass of the orbiting body
  a : float
      Semi-major axis of the orbiting body
  e : float
      Eccentricity of the orbiting body
  inc : float
      Inclination relative to the sky (90 degrees = edge-on/transitting)
      
  Keyword Arguments
  -----------------
  inUnits : string
      Unit system for input parameters . Options are 'iau' (default), 'mks', or 'cgs'
  outUnits : string
      Unit system for return value. Options are 'mks' (default) or 'cgs'
  angUnits : string
      Units for angles. Options are 'deg' (default) or 'rad'
      
  Returns
  -------
  K : float
      Semi-amplitude of the radial velocity
  """
  if inUnits == 'iau':
    pass
  elif inUnits == 'mks':
    m1 /= MSUN
    m2 /= MSUN
    a /= AU
  elif inUnits == 'cgs':
    m1 /= (MSUN*1000)
    m2 /= (MSUN*1000)
    a /= (AU*100)
  else:
    raise ValueError('Invalid units for "inUnits". Valid options are "iau", "mks" or "cgs"')
   
  if angUnits == 'deg':
    inc *= np.pi/180. 
  elif angUnits == 'rad':
    pass
  else:
    raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

  RV = (np.sqrt(G/(1-e**2))*m2*MSUN*np.sin(inc)*(a*AU*(m1+m2)*MSUN)**(-0.5))
  if outUnits == 'mks':
    return RV
  elif outUnits == 'cgs':
    return RV*100
  else:
    raise ValueError('Invalid units for "outUnits". Valid options are "mks" or "cgs"')
    
    
def Per2Semi(ms, m, P, inUnits = 'iau', outUnits = 'iau'):
    """
    Calculates the semi-major axis of a planet/body based on the period
    
    Parameters
    ----------
    ms : float
        The mass of the central object (star)
    m : float
        The mass of the orbiting body
    P : float
        Period of the orbiting body
        
    Keyword Arguments
    -----------------
    inUnits : string
        Unit system for input parameters . Options are 'iau' (default), 'mks', or 'cgs'
    outUnits : string
        Unit system for return value. Options are 'iau' (default), 'mks', or 'cgs'
        
    Returns
    -------
    a : float
        Semi-major axis of orbiting body
    """
    if inUnits == 'iau':
      a = ((P*k)**2.0*(ms+m)/(4*np.pi**2.0))**(1.0/3.0)
    elif inUnits == 'mks':
      a = (P**2*G*(ms+m)/(4*np.pi**2.0))**(1.0/3.0)/AU
    elif inUnits == 'cgs':
      a = (P**2*(G*1000)*(ms+m)/(4*np.pi**2.0))**(1.0/3.0)/(AU*100)
    else:
      raise ValueError('Invalid units for "inUnits". Valid options are "iau", "mks" or "cgs"')
    
    if outUnits == 'iau':
      pass
    elif outUnits == 'mks':
      a *= AU
    elif outUnits == 'cgs':
      a *= AU*100
    else:
      raise ValueError('Invalid units for "outUnits". Valid options are "iau", "mks" or "cgs"')
    
    return a


def Semi2Per(ms, m, a):
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

def LongAscNode(h):
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

def True2EccAnom(f, e):
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

def Osc2X(ms, m, a, e, inc, apc, lasn, manom, inUnits = 'iau', outUnits = 'iau', angUnits = 'deg'):
    #masses in solar units, a in AU, angles in degrees
    if not isinstance(m, np.ndarray):
      m0 = np.array([m])
    else:
      m0 = deepcopy(m)

    if not isinstance(a, np.ndarray):
      a0 = np.array([a])
    else:
      a0 = deepcopy(a)
      
    if not isinstance(e, np.ndarray):
      e0 = np.array([e])
    else:
      e0 = deepcopy(e)

    if not isinstance(inc, np.ndarray):
      inc0 = np.array([inc])
    else:
      inc0 = deepcopy(inc)

    if not isinstance(apc, np.ndarray):
      apc0 = np.array([apc])
    else:
      apc0 = deepcopy(apc)

    if not isinstance(lasn, np.ndarray):
      lasn0 = np.array([lasn])
    else:
      lasn0 = deepcopy(lasn)
  
    if not isinstance(manom, np.ndarray):
      manom0 = np.array([manom])
    else:
      manom0 = deepcopy(manom)
      
    if len(m) != len(a) or len(m) != len(e) or len(m) != len(inc) or len(m) != len(apc) or len(m) != len(lasn) or len(m) != len(manom):
      print ('not all elements have the same length!')
      return None
      
    if inUnits == 'iau':
      pass
    elif inUnits == 'mks':
      m0 /= MSUNkg
      a0 /= AUm
    elif inUnits == 'cgs':
      m0 /= MSUNg
      a0 /= AUc
    else:
      raise ValueError('Invalid units for "inUnits". Valid options are "iau", "mks", or "cgs"')

    if angUnits == 'deg':
      inc0 *= DEG
      apc0 *= DEG
      lasn0 *= DEG
      manom0 *= DEG
    elif angUnits == 'rad':
      pass
    else:
      raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

    nplanets = len(m0)
    xi = np.zeros((2, nplanets))
    vi = np.zeros((2, nplanets))
    eanom = np.zeros(nplanets)
    xastro = np.zeros((3, nplanets))  #nplanets + 1 will be the star
    vastro = np.zeros((3, nplanets))
    
    #----Calc eccentric anomaly and astrocentric x,y,z coords------------
    for j in range(nplanets):
        
        eanom[j] = mech.solve_kepler(manom0[j], e0[j])
                
        xi[0, j] = mech.xinit(a0[j], e0[j], eanom[j])
        xi[1, j] = mech.yinit(a0[j], e0[j], eanom[j])
        
        vi[0, j] = mech.vxi(m0[j], ms, a0[j], xi[0,j], xi[1,j], eanom[j])
        vi[1, j] = mech.vyi(m0[j], ms, a0[j], xi[0,j], xi[1,j], eanom[j], e0[j])
        ang = np.zeros((3,2))
        ang[0] = [mech.xangle1(lasn0[j],apc0[j],inc0[j]), mech.xangle2(lasn0[j],apc0[j],inc0[j])]
        ang[1] = [mech.yangle1(lasn0[j],apc0[j],inc0[j]), mech.yangle2(lasn0[j],apc0[j],inc0[j])]
        ang[2] = [mech.zangle1(apc0[j], inc0[j]), mech.zangle2(apc0[j], inc0[j])]

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


def X2Osc(ms, m, x, v, set_argp = False, argp=np.array([]), set_longa = False, longa=np.array([])):
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
        r = mech.norm(x[:,i])
        vsq = mech.norm(v[:,i]) * mech.norm(v[:,i])
        rdot = sum(x[:,i] * v[:,i]) / r
        mu = k**2.0 * (ms + m[i])
        h = mech.cross(x[:,i], v[:,i])
        hsq = mech.norm(h) * mech.norm(h)

        a[i] = (2.0/r - vsq/mu)**(-1.0)
        e[i] = np.sqrt(1.0 - hsq / (mu*a[i]))
        inc[i] = np.arccos(h[2] / mech.norm(h))
        
        lasn[i] = LongAscNode(h)
        
        if np.isnan(lasn[i]):
          if set_longa == False:
            raise Exception('LongA = NaN, need to set keyword "set_longa"')
          else:
            lasn[i] = longa[i]

        sinwf = x[2,i] / (r*np.sin(inc[i]))
        coswf = (x[0,i]/r + np.sin(lasn[i]) * sinwf * np.cos(inc[i])) / np.cos(lasn[i])
        
        sinf = a[i] * (1.0 - e[i]*e[i]) * rdot / (mech.norm(h) * e[i])
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
        eanom = True2EccAnom(f, e[i])
        manom[i] = eanom - e[i] * np.sin(eanom)

        while manom[i] < 0:
            manom[i] += 2*np.pi
        while manom[i] >= 2*np.pi:
            manom[i] -= 2*np.pi
            
    if len(m) == 1:
      return (a[0], e[0], inc[0]/DEG, apc[0]/DEG, lasn[0]/DEG, manom[0]/DEG)
    else:
      return (a, e, inc/DEG, apc/DEG, lasn/DEG, manom/DEG)
    
def FixAngles(angles, domain = (0,360), angUnits = 'deg'):
  if angUnits == 'deg':
    modulo = 360
    if (domain[1] - domain[0]) != modulo:
      raise ValueError('"domain" must encompass all angles, e.g., (0,360) or (-180,180)')
  elif angUnits == 'rad':
    modulo = 2*np.pi
    if (domain[1] - domain[0]) != modulo:
      raise ValueError('"domain" must encompass all angles, e.g., (0,2*pi) or (-pi,pi)')
  else: 
    raise ValueError('Invalid units for "angUnits". Valid options are "rad" or "deg"')

  if not isinstance(angles, np.ndarray):
    while angles >= domain[1]:
      angles -= modulo
    while angles < domain[0]:
      angles += modulo
      
  else:  
    while np.max(angles) >= domain[1]:
      angles[np.where(angles >= domain[1])] -= modulo
    while np.min(angles) < domain[0]:
      angles[np.where(angles < domain[0])] += modulo
  return angles
    
    
def Astro2Bary(ms, m, xastro, vastro):
  xcom = np.array([[np.sum(m*xastro[0,:])],[np.sum(m*xastro[1,:])],[np.sum(m*xastro[2,:])]])/(np.sum(m)+ms)
  vcom = np.array([[np.sum(m*vastro[0,:])],[np.sum(m*vastro[1,:])],[np.sum(m*vastro[2,:])]])/(np.sum(m)+ms)
  
  xbary = xastro - xcom
  vbary = vastro - vcom
    
  xs = -xcom
  vs = -vcom
  
  return (xbary, vbary, xs, vs)

def AngularM_Osc(ms, m, a, e, inc, apc, lasn, manom):
  xastro, vastro = Osc2X(ms, m, a, e, inc, apc, lasn, manom)
  xbary, vbary, xs, vs = Astro2Bary(ms, m, xastro, vastro)
  
  L = ms * mech.cross(xs, vs)
  for j in range(len(m)):
    L = L + m[j] * mech.cross(xbary[:,j], vbary[:,j])
  
  return L
  
def AngularM_X(ms, m, xastro, vastro):
  xbary, vbary, xs, vs = Astro2Bary(ms, m, xastro, vastro)
  
  L = ms * mech.cross(xs, vs)
  for j in range(len(m)):
    L = L + m[j] * mech.cross(xbary[:,j], vbary[:,j])
  
  return L

def EnergyTot_X(ms, m, xastro, vastro):
  xbary, vbary, xs, vs = Astro2Bary(ms, m, xastro, vastro)
  
  E = 0.5*ms*np.dot(vs.T,vs)
  for j in range(len(m)):
    E += 0.5*m[j]*np.dot(vbary[:,j].T,vbary[:,j])
    E += -k**2*ms*m[j]/np.linalg.norm(xbary[:,j]-xs.T)
    for i in range(j+1,len(m)):
      E += -k**2*m[i]*m[j]/np.linalg.norm(xbary[:,j]-xbary[:,i])
  
  return E
  
def Rotate(theta, phi, m, xv):
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

def InvPlaneX(ms, m, xastro, vastro):
  L = AngularM_X(ms, m, xastro, vastro)
  xbary, vbary, xs, vs = Astro2Bary(ms, m, xastro, vastro)
  
  theta = np.arctan2(L[1],L[0])
  phi = np.arctan2(np.sqrt(L[1]**2.0+L[0]**2.0),L[2])
  
  xinv = Rotate(theta, phi, m, xbary)
  vinv = Rotate(theta, phi, m, vbary)
  xsi = Rotate(theta, phi, ms, xs)
  vsi = Rotate(theta, phi, ms, vs)
  xsi = np.array([[xsi[0]], [xsi[1]], [xsi[2]]])
  vsi = np.array([[vsi[0]], [vsi[1]], [vsi[2]]])
  
  return xinv, vinv, xsi, vsi, theta, phi


def AstroInvPlaneX(ms, m, xastro, vastro):
  xinv, vinv, xsi, vsi, theta, phi = InvPlaneX(ms, m, xastro, vastro)
  xai = xinv - xsi
  vai = vinv - vsi
  
  return xai, vai
  

def AngularM_Sec(ms, m, a, e, inc, longa):
  hz = np.sqrt((k**2.0*(ms+m))*a*(1.0-e**2.0))  #angular m/mass of planets in own orbital planes
  hX = hz*np.sin(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hY = -hz*np.cos(longa*np.pi/180.0)*np.sin(inc*np.pi/180.0)
  hZ = hz*np.cos(inc*np.pi/180.0)
  
  Lx = m * hX
  Ly = m * hY
  Lz = m * hZ
  
  L = np.array([np.sum(Lx), np.sum(Ly), np.sum(Lz)])
  return L
  
def Hill_Gladman(ms, m, a, e):
  mu1 = m[0]/ms
  mu2 = m[1]/ms
  alpha = mu1+mu2
  gam1 = (1.0-e[0]**2.)**0.5
  gam2 = (1.0-e[1]**2.)**0.5
  delta = (a[1]/a[0])**0.5
  
  lhs = alpha**(-3.)*(mu1+mu2/delta**2.)*(mu1*gam1+mu2*gam2*delta)**2.0
  rhs = 1.0+3.0**(4./3)*mu1*mu2/alpha**(4./3)
  
  return lhs/rhs
  
def Hill_MB(ms, m, a, e, inc, argp, longa, meana):
  xa, va = Osc2X(ms,m,a,e,inc,argp,longa,meana)
  L = AngularM_X(ms,m,xa,va)
  L2 = np.dot(L.T,L)

  E = EnergyTot_X(ms,m,xa,va)
  
  Mstar = (ms*m[0]+ms*m[1]+m[0]*m[1])
  lhs = -2*(ms+m[0]+m[1])*L2*E/(k**4*Mstar**3)
  rhs = 1.0+3.0**(4./3)*m[0]*m[1]/(ms**(2./3)*(m[0]+m[1])**(4./3)) - \
        m[0]*m[1]*(11.*m[0]+7.*m[1])/(3.*ms*(m[0]+m[1])**2.)
  return lhs/rhs

def Ecc2TrueAnom(eanom, e):
# calculate true anomaly from eccentric anomaly (radians) and eccentricity
# will not work if e >= 1 !!!!!
  tanf2 = np.sqrt((1+e)/(1-e))*np.tan(eanom/2.0)
  
  f = 2.0*np.arctan(tanf2)
  
  if f < 0:
    f += 2*np.pi
    
  return f

def MeanL2TrueA(e, argp, meanl):
# input argp and meanl in degrees
  meana = (meanl - argp)%360
  ecca = solve_kepler(meana*np.pi/180.,e)  #ecca will be in radians
  f = ecc2true_anom(ecca, e)*180./np.pi  #get true anom and convert to degrees
  
  return f #return true anomaly in degrees  
  
def RadialVel(ms,mp,P,e,argp,inc,meanl,planet=False):
#velocity in m/s of star due to perturber along line of sight
  truea = meanl2truea(e,argp,meanl)
  a = ((P*daysec)**2 * G *(ms+mp)/(4*np.pi**2))**(1./3)
  return -np.sqrt(G/((ms+mp)*(a*(1-e**2))))*mp*np.sin(inc*np.pi/180.)\
        *(np.cos((argp+truea)*np.pi/180.)+e*np.cos(argp*np.pi/180.))

