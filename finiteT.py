"""
This module provides the functions for the one-loop finite
temperature corrections to a potential in QFT.  The two basic
functions are:

    Jb(x) = int[0->inf] dy +y^2 log( 1 - exp(-sqrt(x^2 + y^2)) )

    Jf(x) = int[0->inf] dy -y^2 log( 1 + exp(-sqrt(x^2 + y^2)) )

Call them by:

    Jb(x, approx='high', deriv=0, n = 8)

Here, approx can either be 'exact', 'spline', 'high', or 'low'.
Exact calculates the integral numerically, while high and low
calculate the high and low x expansions of J to order n.
Specify the derivative with the 'deriv' parameter.

Adapted from cosmoTransitions.finiteT by Carroll L. Wainwright:
https://github.com/clwainwright/CosmoTransitions

MIT License

Copyright (c) 2012-2017 Carroll L. Wainwright

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import numpy
from scipy import integrate, interpolate
from scipy import special
from scipy.special import factorial as fac

pi = numpy.pi
euler_gamma = 0.577215661901532
log, exp, sqrt = numpy.log, numpy.exp, numpy.sqrt
array = numpy.array

spline_data_path = os.path.dirname(__file__)


# The following are the exact integrals:

def _Jf_exact(x):
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(abs(x*x)-y*y)/2)))
        return (
            integrate.quad(f1,0,abs(x))[0] +
            integrate.quad(f,abs(x),numpy.inf)[0]
        )


def _Jf_exact2(theta):
    # Note that this is a function of theta so that you can get negative values
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(-theta-y*y)/2)))
        num_domains = int(numpy.ceil(0.5*(1.+(sqrt(abs(theta))/pi))))
        n_itn = numpy.arange(0, num_domains, 1)
        x_pos = sqrt(-(2.*n_itn-1.)**2*pi**2 - theta)
        x_pos[0] = sqrt(abs(theta))
        x_pos = numpy.append(x_pos, 0.)
        integral = 0.
        for n in n_itn:
            integral += integrate.quad(f1, x_pos[n+1], x_pos[n])[0]
        return (
            integrate.quad(f, abs(theta)**.5, numpy.inf)[0] +
            integral
            #integrate.quad(f1, 0, abs(theta)**.5)[0]
        )


def _Jb_exact(x):
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(abs(x*x)-y*y)/2)))
        return (
            integrate.quad(f1,0,abs(x))[0] +
            integrate.quad(f,abs(x),numpy.inf)[0]
        )


def _Jb_exact2(theta):
    # Note that this is a function of theta so that you can get negative values
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(-theta-y*y)/2)))
        num_domains = int(numpy.ceil(sqrt(abs(theta))/(2.*pi)))
        n_itn = numpy.arange(0, num_domains, 1)
        x_pos = sqrt(-4.*n_itn**2*pi**2 - theta)
        x_pos = numpy.append(x_pos, 0.)
        integral = 0.
        for n in n_itn:
            integral += integrate.quad(f1, x_pos[n+1], x_pos[n])[0]
        
        return (
            integrate.quad(f, abs(theta)**.5, numpy.inf)[0] +
            integral
            #integrate.quad(f1, 0, abs(theta)**.5)[0]
        )


def _dJf_exact(x):
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))+1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]


def _dJb_exact(x):
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))-1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]


def arrayFunc(f, x, typ=float):
    return numpy.vectorize(f, otypes=[typ])(x)


def Jf_exact(x):
    """Jf calculated directly from the integral."""
    return arrayFunc(_Jf_exact, x, complex)


def Jf_exact2(theta):
    """Jf calculated directly form the integral; input is theta = x^2."""
    return arrayFunc(_Jf_exact2, theta)


def Jb_exact(x):
    """Jb calculated directly from the integral."""
    return arrayFunc(_Jb_exact, x)


def Jb_exact2(theta):
    """Jb calculated directly form the integral; input is theta = x^2."""
    return arrayFunc(_Jb_exact2, theta)


def dJf_exact(x):
    """dJf/dx calculated directly from the integral."""
    return arrayFunc(_dJf_exact, x)


def dJb_exact(x):
    """dJb/dx calculated directly from the integral."""
    return arrayFunc(_dJb_exact, x)


# Spline fitting. The bundled tables use linear grids at large negative x,
# followed by a dense sinh grid around x = 0.
_xfmin = -1000.
_xfmax = 1.35e3
_xbmin = -25000.
_xbmax = 1.41e3
_Jf_dat_path = spline_data_path+"/finiteT_f.dat.txt"
_Jb_dat_path = spline_data_path+"/finiteT_b.dat.txt"


def canonical_jf_grid():
    fine_min = -6.82200203
    fine = abs(fine_min) * numpy.sinh(
        numpy.linspace(
            numpy.arcsinh(-1.3 * 20),
            numpy.arcsinh(-20 * _xfmax / fine_min),
            1000,
        )
    ) / 20
    return numpy.concatenate((numpy.arange(-1000.0, -8.0, 1.0), fine))


def canonical_jb_grid():
    fine_min = -3.72402637
    fine = abs(fine_min) * numpy.sinh(
        numpy.linspace(
            numpy.arcsinh(-1.3 * 20),
            numpy.arcsinh(-20 * _xbmax / fine_min),
            1000,
        )
    ) / 20
    return numpy.concatenate(
        (
            numpy.arange(-25000.0, -4990.0, 10.0),
            numpy.arange(-4998.0, -998.0, 2.0),
            numpy.arange(-999.0, -4.0, 1.0),
            fine,
            numpy.array([1500.0]),
        )
    )


def generate_spline_tables(output_dir=spline_data_path):
    """Generate the canonical spline tables outside the import path."""
    paths = (
        os.path.join(output_dir, "finiteT_f.dat.txt"),
        os.path.join(output_dir, "finiteT_b.dat.txt"),
    )
    os.makedirs(output_dir, exist_ok=True)
    xf = canonical_jf_grid()
    xb = canonical_jb_grid()
    numpy.savetxt(paths[0], numpy.column_stack((xf, Jf_exact2(xf))), fmt="%.8e")
    numpy.savetxt(paths[1], numpy.column_stack((xb, Jb_exact2(xb))), fmt="%.8e")
    return paths


_xf, _yf = numpy.loadtxt(_Jf_dat_path).T


_tckf = interpolate.splrep(_xf, _yf)
def Jf_spline(X,n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x,_tckf, der=n).ravel()
    y[x < _xfmin] = interpolate.splev(_xfmin,_tckf, der=n)
    y[x > _xfmax] = 0
    return y.reshape(X.shape)


_xb, _yb = numpy.loadtxt(_Jb_dat_path).T


_tckb = interpolate.splrep(_xb, _yb)
def Jb_spline(X,n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x,_tckb, der=n).ravel()
    y[x < _xbmin] = interpolate.splev(_xbmin,_tckb, der=n)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)


# Now for the low x expansion (require that n <= 50)
a,b,c,d = -pi**4/45, pi*pi/12, -pi/6, -1/32.
logab = 1.5 - 2*euler_gamma + 2*log(4*pi)
l = numpy.arange(50)+1
g = (-2*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1)) *
     special.gamma(l+.5)/(fac(l+2)*(2*pi)**(2*l+4)))
lowCoef_b = (a,b,c,d,logab,l,g)
del (a,b,c,d,logab,l,g)  # clean up name space

a,b,d = -7*pi**4/360, pi*pi/24, 1/32.
logaf = 1.5 - 2*euler_gamma + 2*log(pi)
l = numpy.arange(50)+1
g = (.25*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1)) *
     special.gamma(l+.5)*(1-.5**(2*l+1))/(fac(l+2)*pi**(2*l+4)))
lowCoef_f = (a,b,d,logaf,l,g)
del (a,b,d,logaf,l,g)  # clean up name space


def Jb_low(x,n=20):
    """Jb calculated using the low-x (high-T) expansion."""
    (a,b,c,d,logab,l,g) = lowCoef_b
    y = a + x*x*(b + x*(c + d*x*(numpy.nan_to_num(log(x*x)) - logab)))
    for i in range(1, n + 1):
        y += g[i-1]*x**(2*i+4)
    return y


def Jf_low(x,n=20):
    """Jf calculated using the low-x (high-T) expansion."""
    (a,b,d,logaf,l,g) = lowCoef_f
    y = a + x*x*(b + d*x*x*(numpy.nan_to_num(log(x*x)) - logaf))
    for i in range(1, n + 1):
        y += g[i-1]*x**(2*i+4)
    return y


# The next few functions are all for the high approximation

def x2K2(k,x):
    y = -x*x*special.kn(2, k*x)/(k*k)
    return numpy.where(x == 0, -2.0/k**4, y)


def dx2K2(k,x):
    y = abs(x)
    return numpy.nan_to_num(x*y*special.kn(1,k*y)/k)


def d2x2K2(k,x):
    x = abs(x)
    y = numpy.nan_to_num(x*(special.kn(1,k*x)/k - x*special.kn(0,k*x)))
    return numpy.where(x == 0, 1.0/k**2, y)


def d3x2K2(k,x):
    y = abs(x)
    return numpy.nan_to_num(x*(y*k*special.kn(1,k*y) - 3*special.kn(0,k*y)))


def Jb_high(x, deriv=0, n=8):
    """Jb calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y = 0.0
    for k in range(1, n + 1):
        y += K(k,x)
    return y


def Jf_high(x, deriv=0, n=8):
    """Jf calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y = 0.0
    for k in range(1, n + 1):
        y += (-1)**(k-1)*K(k,x)
    return y


# And here are the final functions:
# Note that if approx = 'spline', the function called is
# J(theta) (x^2 -> theta so you can get negative mass squared)
def Jb(x, approx='high', deriv=0, n=8):
    """
    A shorthand for calling one of the Jb functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative).
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """
    if(approx == 'exact'):
        return (Jb_exact, dJb_exact)[deriv](x)
    elif(approx == 'spline'):
        return Jb_spline(x, deriv)
    elif(approx == 'low'):
        return Jb_low(x,n)
    elif(approx == 'high'):
        return Jb_high(x, deriv, n)


def Jf(x, approx='high', deriv=0, n=8):
    """
    A shorthand for calling one of the Jf functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative).
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """
    if(approx == 'exact'):
        return (Jf_exact, dJf_exact)[deriv](x)
    elif(approx == 'spline'):
        return Jf_spline(x, deriv)
    elif(approx == 'low'):
        return Jf_low(x,n)
    elif(approx == 'high'):
        return Jf_high(x, deriv, n)
