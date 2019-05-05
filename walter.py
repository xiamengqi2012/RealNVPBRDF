
import numpy as np

# Walter BSDF evaluation and sampling.
# Implementation ported from RIS walterbxdf.h and support files

# All these functions work for arrays.

# indexing expression to add a singleton dimension at the end
nax = (..., np.newaxis)

def SQRT(x): return np.sqrt(x)
def MAX(a,b): return np.maximum(a, b)
def SINCOS(t): return (np.sin(t), np.cos(t))
def ABS(x): return np.abs(x)
def SIGN(x): return np.sign(x)

def makeRtVector3(x, y, z):
    return np.stack((x, y, z), -1)

def makeRtColorRGB(c):
    return np.stack((c, c, c), -1)

def Dot(u, v):
    return (u * v).sum(-1)

def Max(u):
    return np.amax(u, -1)

def Normalize(u):
    return u / np.sqrt(Dot(u, u))[...,np.newaxis]

def SphericalDirection( sintheta, costheta, phi ):
    sinphi, cosphi = SINCOS( phi );
    return makeRtVector3( sintheta * cosphi, sintheta * sinphi, costheta )

def ReflectedVector( V, H, VdotH ):
    return 2.0 * VdotH[...,np.newaxis] * H - V

def RefractedVector( V, H, VdotH, VdotN, eta ):
    # eta is a scalar
    ieta = 1.0 / eta
    coef = VdotH * ieta - SIGN( VdotN ) * SQRT( 1.0 + ( VdotH*VdotH-1.0 ) * ( ieta*ieta ) )

    res = coef[...,np.newaxis] * H - V*ieta
    Normalize( res )
    return res

def fresnel(n_i, n_t, mu_i):
    lm_t2 = (n_i / n_t)**2 * (1 - mu_i**2)
    mu_t = np.sqrt(1 - np.minimum(lm_t2, 1))
    R_s = ((n_i * mu_i - n_t * mu_t) / (n_i * mu_i + n_t * mu_t))**2
    R_p = ((n_i * mu_t - n_t * mu_i) / (n_i * mu_t + n_t * mu_i))**2
    return np.select([mu_i > 0], [(R_s + R_p) / 2], 1.0)

def Fresnel( VdotH, eta ):
    return fresnel(1.0, eta, VdotH)

def chi_plus(x):
    return np.where(x > 0, 1, 0)

def sampleH(roughness2, xi0, xi1, dist):
    if dist == 'G':
        # GGX
        # Sample angle theta: eq 35
        tantheta2 = roughness2 * xi0 / ( 1.0 - xi0 )
        costheta2 = 1.0 / ( 1.0 + tantheta2 )
        costheta  = SQRT( costheta2 )
        sintheta  = SQRT( MAX( 0.0, 1.0 - costheta2 ) )
    elif dist == 'B':
        # Beckmann
        # sample angle theta: eq 28
        tantheta2 = -roughness2 * np.log(1-xi0)
        costheta2 = 1.0 / ( 1.0 + tantheta2 )
        costheta  = SQRT( costheta2 )
        sintheta  = SQRT( MAX( 0.0, 1.0 - costheta2 ) )
    else:
        raise ValueError('dist is neither G nor B')
    
    return (SphericalDirection( sintheta, costheta, 2 * np.pi * xi1 ), costheta)

def BeckmannG1(value, a):
    return chi_plus(value) * (3.535*a + 2.181*a*a)/(1 + 2.276*a + 2.577*a*a)

def BeckmannG2(value):
    return chi_plus(value)

def evaluate(roughness, eta, wo, wi, dist):
    # return brdf value (didn't multiply cos)
    """Evaluate BRDF and PDFs for Walter BxDF."""

    # eta is assumed > 1, and it is the refractive index of the side of the
    # surface facing away from the normal.

    rGain = 1.0
    boostReflect = 1.0
    tAlbedo = np.array((1.0, 1.0, 1.0))

    # Convention is for forward path tracing; "i" is the viewer and "o" is the light.
    VdotN = wi[...,2]
    LdotN  = wo[...,2]

    isRefraction = ( LdotN*VdotN < 0.0 )

    # Refractive indices for the two sides.  eta_o is the index for the side
    # opposite wi, even when wo is on the same side.
    eta_i = np.where(VdotN > 0, 1.0, eta)
    eta_o = np.where(VdotN > 0, eta, 1.0)

    # Half vector.
    H = np.where(isRefraction[nax],
                 -(eta_o[nax] * wo + eta_i[nax] * wi),
                 SIGN(VdotN)[nax] * (wo + wi))
    H = Normalize( H )
    # check side for H
    correct = H[...,2] * wi[...,2] > 0

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )

    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )

    # This seems always to compute Fresnel factor for the ray coming from outside.
    #F = Fresnel( absVdotH, eta )

    # Compute fresnel factor for the appropriate side of the surface.
    F = fresnel(eta_i, eta_o, absVdotH)

    chooseReflect = F * rGain * boostReflect
    chooseRefract = (1.0-F) * Max( tAlbedo )
    total = chooseRefract + chooseReflect

    chooseReflect = chooseReflect / total
    chooseRefract = 1.0 - chooseReflect

    roughness2 = roughness*roughness
    HdotN =  H[...,2]
    costheta  = ABS( HdotN )
    costheta2 = costheta * costheta

    if dist == 'G':
        # Compute the microfacet distribution (GGX): eq 33
        alpha2_tantheta2 = roughness2 + ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )

        # Compute the Smith shadowing terms: eq 34 and eq 23
        LdotN2 = LdotN * LdotN
        VdotN2 = VdotN * VdotN

        iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
        iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )
        G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0 / ( iG1o * iG1i )
    elif dist == 'B':
        # Beckmann distribution and shadowing masking term
        # Compute the Beckmann Distribution: eq 25
        tantheta2 = ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN)/(np.pi * roughness2 * costheta2 * costheta2) * np.exp(-tantheta2/roughness2);
        
        # Shadowing masking term for Beckmann: eq 27
        costhetav = ABS(VdotN)
        tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
        a = 1.0/(roughness * tanthetav)
        iG1i = np.where(a<1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))
        
        costhetal = ABS(LdotN)
        tanthetal = SQRT(1 - costhetal*costhetal)/costhetal
        a = 1.0/(roughness * tanthetal)
        iG1o = np.where(a<1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))
        G = iG1i * iG1o
    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41

    # Refraction case
    denom = ( VdotH + (eta_o/eta_i) * LdotH)**2
    idenom = 1.0 / denom
    fJacobian = absLdotH * idenom
    rJacobian = absVdotH * idenom

    #    refract_value = tAlbedo * ( (1.0-F) * D * G * absVdotH * fJacobian * (eta_o/eta_i)**2 / ABS( VdotN ) )[...,np.newaxis] # baking LdotN
    refract_value = tAlbedo * ( (1.0-F) * D * G * absVdotH * fJacobian * (eta_o/eta_i)**2 / (ABS( VdotN ) *ABS( LdotN )))[...,np.newaxis] # not baking LdotN
    refract_fpdf = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * chooseRefract * D*costheta * fJacobian * (eta_o/eta_i)**2
    refract_rpdf = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * chooseRefract * D*costheta * rJacobian


    # Reflection case
    jacobian = 1.0 / ( 4.0 * absLdotH ) # LdotH = VdotH by definition

    # reflect_value = makeRtColorRGB( rGain * F * D * G / ( 4.0 * ABS( VdotN ) ) ) # baking LdotN
    reflect_value = makeRtColorRGB( rGain * F * D * G / ( 4.0 * ABS( VdotN ) * ABS( LdotN )) ) # no baking LdotN
    reflect_fpdf = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * chooseReflect * D*costheta * jacobian
    reflect_rpdf = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * chooseReflect * D*costheta * jacobian


    value = np.where(isRefraction[...,np.newaxis], refract_value, reflect_value)
    fpdf = np.where(isRefraction, refract_fpdf, reflect_fpdf)
    rpdf = np.where(isRefraction, refract_rpdf, reflect_rpdf)

    return  (value, fpdf, rpdf )


def sample(xi0, xi1, xi2, roughness, eta, wi, dist):
    """Generate a sample from the Walter BxDF by sampling normal distribution."""
    """ TODO: Implement sampling according to visible normal"""

    rGain = 1.0
    boostReflect = 1.0
    tAlbedo = np.array((1.0, 1.0, 1.0))
    roughness2 = roughness * roughness
    
    (H, costheta) = sampleH(roughness2, xi0, xi1, dist)

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )

    VdotN = wi[...,2]
    eta_i = np.where(VdotN > 0, 1.0, eta)
    eta_o = np.where(VdotN > 0, eta, 1.0)
    F = fresnel(eta_i, eta_o, absVdotH)

    chooseReflect = F * rGain * boostReflect;
    chooseRefract = (1.0-F) * Max( tAlbedo );
    total = chooseRefract + chooseReflect;

    chooseReflect = chooseReflect / total;
    chooseRefract = 1.0 - chooseReflect;

    # choose reflection or refraction
    doRefraction = (xi2 >= chooseReflect)
    if VdotN[0] > 0:
        e = eta
    else:
        e = 1/eta
    wo = np.where(doRefraction[...,np.newaxis],
        RefractedVector( wi, H, VdotH, VdotN, e ),
        ReflectedVector( wi, H, VdotH )
    )

    HdotN =  H[...,2]
    LdotN = wo[...,2]
    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )
                                   
    if dist == 'G':
       # Compute the microfacet distribution (GGX): eq 33
       tantheta2 = roughness2 * xi0 / ( 1.0 - xi0 )
       costheta2 = 1.0 / ( 1.0 + tantheta2 )
       alpha2_tantheta2 = roughness2 + tantheta2
       D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )
       
       # Compute the Smith shadowing terms: eq 34 and eq 23
       LdotN  = wo[...,2]
       LdotN2 = LdotN * LdotN
       VdotN2 = VdotN * VdotN
       
       iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
       iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )
       
       LdotH = Dot( wo, H )
       LdotN = wo[...,2]
       G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0/( iG1o * iG1i )
    elif dist == 'B':
       # Beckmann distribution and shadowing masking term
       # Compute the Beckmann Distribution: eq 25
       costh = ABS(HdotN)
       costh2 = costh**2
       tanth2 = ( 1.0 - costh2 ) / costh2
       D = chi_plus(HdotN)/(np.pi * roughness2 * costh2 * costh2) * np.exp(-tanth2/roughness2);
       # Shadowing masking term for Beckmann: eq 27
       costhetav = ABS(VdotN)
       tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
       a = 1.0/(roughness * tanthetav)
       
       iG1i = np.where(a<1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))
       
       costhetal = ABS(LdotN)
       tanthetal = SQRT(1 - costhetal*costhetal)/costhetal
       
       a = 1.0/(roughness * tanthetal)
       iG1o = np.where(a<1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))

       G = iG1i * iG1o
   
    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41
    # Refraction case
    denom = (VdotH + eta_o/eta_i * LdotH)**2
    idenom = 1.0 / denom
    fJacobian = absLdotH * idenom
    rJacobian = absVdotH * idenom

    # side check
    correct_refract = wi[...,2] * wo[...,2] <0
    refract_value = np.where(correct_refract[nax], tAlbedo * ( (1.0-F) * D * G * absVdotH * fJacobian * (eta_o/eta_i)**2 / (ABS( VdotN ) *ABS( LdotN )))[nax], makeRtColorRGB(0.0)) # not baking LdotN
    refract_fpdf = np.where(correct_refract, chooseRefract * D * costheta * fJacobian * (eta_o/eta_i)**2, 0)
    refract_rpdf = np.where(correct_refract, chooseRefract * D * costheta * rJacobian, 0)
    
    # Reflection case
    jacobian = 1.0 / ( 4.0 * absLdotH ) # LdotH = VdotH by definition

    # side check
    correct_reflect = wi[...,2] * wo[...,2] >0
    reflect_value = np.where(correct_reflect[nax], makeRtColorRGB( rGain * F * D * G / ( 4.0 * ABS( VdotN ) * ABS( LdotN )) ), makeRtColorRGB(0.0)) # baking LdotN
    reflect_fpdf = np.where(correct_reflect, chooseReflect * D * costheta * jacobian, 0);
    reflect_rpdf = np.where(correct_reflect, chooseReflect * D * costheta * jacobian, 0);

    value = np.where(doRefraction[nax], refract_value, reflect_value)
    fpdf = np.where(doRefraction, refract_fpdf, reflect_fpdf)
    rpdf = np.where(doRefraction, refract_rpdf, reflect_rpdf)

    return (wo, value, fpdf, rpdf)

def evaluate_reflect(roughness, eta, wo, wi, dist):
    # return brdf value (didn't multiply cos), no fresnel term
    # retrun pdf, no fresnel term
    """Evaluate BRDF and PDFs for Walter BxDF."""

    # eta is assumed > 1, and it is the refractive index of the side of the
    # surface facing away from the normal.

    # Convention is for forward path tracing; "i" is the viewer and "o" is the light.
    VdotN = wi[...,2]
    LdotN  = wo[...,2]

    # Half vector.
    H = SIGN(VdotN)[nax] * (wo + wi)
    H = Normalize( H )
    # check side for H
    correct = (H[...,2] * wi[...,2] * wi[...,2]> 0) & (LdotN*VdotN > 0.0)

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )

    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )
    
    if eta==0:
        F=1
    else:
        eta_i = np.where(VdotN > 0, 1, eta)
        eta_o = np.where(VdotN > 0, eta, 1.0)
        F = fresnel(eta_i, eta_o, absVdotH)

    roughness2 = roughness*roughness
    HdotN =  H[...,2]
    costheta  = ABS( HdotN )
    costheta2 = costheta * costheta

    if dist == 'G':
        # Compute the microfacet distribution (GGX): eq 33
        alpha2_tantheta2 = roughness2 + ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )

        # Compute the Smith shadowing terms: eq 34 and eq 23
        LdotN2 = LdotN * LdotN
        VdotN2 = VdotN * VdotN

        iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
        iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )
        G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0 / ( iG1o * iG1i )
    elif dist == 'B':
        # Beckmann distribution and shadowing masking term
        # Compute the Beckmann Distribution: eq 25
        tantheta2 = ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN)/(np.pi * roughness2 * costheta2 * costheta2) * np.exp(-tantheta2/roughness2);
        # Shadowing masking term for Beckmann: eq 27
        costhetav = ABS(VdotN)
        tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
        a = 1.0/(roughness * tanthetav)
        iG1i = np.where(a<1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))
        
        costhetal = ABS(LdotN)
        tanthetal = SQRT(1 - costhetal*costhetal)/costhetal
        a = 1.0/(roughness * tanthetal)
        iG1o = np.where(a<1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))
        G = iG1i * iG1o
    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41
    # Reflection case
    jacobian = 1.0 / ( 4.0 * absLdotH ) # LdotH = VdotH by definition

    value = np.where(correct[nax], makeRtColorRGB( F * D * G / ( 4.0 * ABS( VdotN ) * ABS( LdotN )) ), makeRtColorRGB(0.0)) # no baking LdotN
#    value = np.where(correct[nax], makeRtColorRGB( F), makeRtColorRGB(0.0)) # no baking LdotN
    fpdf = np.where(correct, chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) *D*costheta * jacobian, 0)
    rpdf = np.where(correct, chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) *D*costheta * jacobian, 0)

    return  (value, fpdf, rpdf)


def evaluate_refract(roughness, eta, wo, wi, dist):
    
    # return brdf value (didn't multiply cos), no fresnel term
    # retrun pdf, no fresnel term
    """Evaluate BRDF and PDFs for Walter BxDF."""
    tAlbedo = np.array((1.0, 1.0, 1.0))

    # eta is assumed > 1, and it is the refractive index of the side of the
    # surface facing away from the normal.

    # Convention is for forward path tracing; "i" is the viewer and "o" is the light.
    VdotN = wi[...,2]
    LdotN  = wo[...,2]

    # Refractive indices for the two sides.  eta_o is the index for the side
    # opposite wi, even when wo is on the same side.
    eta_i = np.where(VdotN > 0, 1.0, eta)
    eta_o = np.where(VdotN > 0, eta, 1.0)

    # Half vector.
    H = -(eta_o[nax] * wo + eta_i[nax] * wi)
    H = Normalize( H )
    # check side
#    correct = (H[...,2] * wi[...,2] > 0.0) & (LdotN*VdotN <0.0)
    correct = LdotN*VdotN <0.0

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )

    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )
    
    F = fresnel(eta_i, eta_o, absVdotH)

    roughness2 = roughness*roughness
    HdotN =  H[...,2]
    costheta  = ABS( HdotN )
    costheta2 = costheta * costheta

    if dist == 'G':
        # Compute the microfacet distribution (GGX): eq 33
        alpha2_tantheta2 = roughness2 + ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )

        # Compute the Smith shadowing terms: eq 34 and eq 23
        LdotN2 = LdotN * LdotN
        VdotN2 = VdotN * VdotN

        iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
        iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )
        G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0 / ( iG1o * iG1i )
    elif dist == 'B':
        # Beckmann distribution and shadowing masking term
        # Compute the Beckmann Distribution: eq 25
        tantheta2 = ( 1.0 - costheta2 ) / costheta2
        D = chi_plus(HdotN)/(np.pi * roughness2 * costheta2 * costheta2) * np.exp(-tantheta2/roughness2);
        # Shadowing masking term for Beckmann: eq 27
        costhetav = ABS(VdotN)
        tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
        a = 1.0/(roughness * tanthetav)
        iG1i = np.where(a<1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))

        costhetal = ABS(LdotN)
        tanthetal = SQRT(1 - costhetal*costhetal)/costhetal
        a = 1.0/(roughness * tanthetal)
        iG1o = np.where(a<1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))
        G = iG1i * iG1o
    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41

    # Refraction case
    denom = ( VdotH + (eta_o/eta_i) * LdotH)**2
    idenom = 1.0 / denom
    fJacobian = absLdotH * idenom
    rJacobian = absVdotH * idenom

    value = np.where(correct[nax], tAlbedo * ( (1-F) * D * G * absVdotH * fJacobian * (eta_o/eta_i)**2 / ( ABS( VdotN) * ABS( LdotN )))[nax], makeRtColorRGB(0.0)) # not baking LdotN
    fpdf = np.where(correct, chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * D*costheta * fJacobian * (eta_o/eta_i)**2, 0)
    rpdf = np.where(correct, chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * D*costheta * rJacobian, 0)

    return  (value, fpdf, rpdf)



def sample_reflect(xi0, xi1, roughness, eta, wi, dist):
    """Generate a sample from the Walter BxDF by sampling normal distribution."""
    """ TODO: Implement sampling according to visible normal"""

    roughness2 = roughness * roughness

    (H, costheta) = sampleH(roughness2, xi0, xi1, dist)

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )
    VdotN = wi[...,2]
    wo = ReflectedVector( wi, H, VdotH )

    HdotN =  H[...,2]
    LdotN = wo[...,2]
    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )
    
    if eta==0:
        F=1
    else:
        eta_i = np.where(VdotN > 0, 1, eta)
        eta_o = np.where(VdotN > 0, eta, 1.0)
        F = fresnel(eta_i, eta_o, absVdotH)

    if dist == 'G':
       # Compute the microfacet distribution (GGX): eq 33
       tantheta2 = roughness2 * xi0 / ( 1.0 - xi0 )
       costheta2 = 1.0 / ( 1.0 + tantheta2 )
       alpha2_tantheta2 = roughness2 + tantheta2
       D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )

       # Compute the Smith shadowing terms: eq 34 and eq 23
       LdotN  = wo[...,2]
       LdotN2 = LdotN * LdotN
       VdotN2 = VdotN * VdotN

       iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
       iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )

       LdotH = Dot( wo, H )
       LdotN = wo[...,2]
       G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0/( iG1o * iG1i )
    elif dist == 'B':
       # Beckmann distribution and shadowing masking term
       # Compute the Beckmann Distribution: eq 25
       costh = ABS(HdotN)
       costh2 = costh**2
       tanth2 = ( 1.0 - costh2 ) / costh2
       D = chi_plus(HdotN)/(np.pi * roughness2 * costh2 * costh2) * np.exp(-tanth2/roughness2);
       # Shadowing masking term for Beckmann: eq 27
       costhetav = ABS(VdotN)
       tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
       a = 1.0/(roughness * tanthetav)

       iG1i = np.where(a<1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))

       costhetal = ABS(LdotN)
       tanthetal = SQRT(1 - costhetal*costhetal)/costhetal

       a = 1.0/(roughness * tanthetal)
       iG1o = np.where(a<1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))

       G = iG1i * iG1o

    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41
    # Reflection case
    jacobian = 1.0 / ( 4.0 * absLdotH ) # LdotH = VdotH by definition
    # side check
    correct_reflect = wi[...,2] * wo[...,2] >0
    value = np.where(correct_reflect[nax], makeRtColorRGB( F * D * G / ( 4.0 * ABS( VdotN ) * ABS( LdotN )) ), makeRtColorRGB(0.0)) # baking LdotN
    fpdf = np.where(correct_reflect, D * costheta * jacobian, 0)
    rpdf = np.where(correct_reflect, D * costheta * jacobian, 0)

    return (wo, value, fpdf, rpdf)

def sample_refract(xi0, xi1, roughness, eta, wi, dist):
    """Generate a sample from the Walter BxDF by sampling normal distribution."""
    """ TODO: Implement sampling according to visible normal"""

    tAlbedo = np.array((1.0, 1.0, 1.0))
    roughness2 = roughness * roughness

    (H, costheta) = sampleH(roughness2, xi0, xi1, dist)

    VdotH = Dot( wi, H )
    absVdotH = ABS( VdotH )

    VdotN = wi[...,2]
    eta_i = np.where(VdotN > 0, 1.0, eta)
    eta_o = np.where(VdotN > 0, eta, 1.0)

    if VdotN[0] > 0:
        e = eta
    else:
        e = 1/eta
    wo = RefractedVector( wi, H, VdotH, VdotN, e )

    HdotN =  H[...,2]
    LdotN = wo[...,2]
    LdotH = Dot( wo, H )
    absLdotH = ABS( LdotH )

    F = fresnel(eta_i, eta_o, absVdotH)

    if dist == 'G':
       # Compute the microfacet distribution (GGX): eq 33
       tantheta2 = roughness2 * xi0 / ( 1.0 - xi0 )
       costheta2 = 1.0 / ( 1.0 + tantheta2 )
       alpha2_tantheta2 = roughness2 + tantheta2
       D = chi_plus(HdotN) * roughness2 / np.pi / ( costheta2*costheta2 * alpha2_tantheta2*alpha2_tantheta2 )

       # Compute the Smith shadowing terms: eq 34 and eq 23
       LdotN  = wo[...,2]
       LdotN2 = LdotN * LdotN
       VdotN2 = VdotN * VdotN

       iG1o = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - LdotN2 ) / LdotN2 )
       iG1i = 1.0 + SQRT( 1.0 + roughness2 * ( 1.0 - VdotN2 ) / VdotN2 )

       LdotH = Dot( wo, H )
       LdotN = wo[...,2]
       G = chi_plus(VdotH/VdotN) * chi_plus(LdotH/LdotN) * 4.0/( iG1o * iG1i )
    elif dist == 'B':
       # Beckmann distribution and shadowing masking term
       # Compute the Beckmann Distribution: eq 25
       costh = ABS(HdotN)
       costh2 = costh**2
       tanth2 = ( 1.0 - costh2 ) / costh2
       D = chi_plus(HdotN)/(np.pi * roughness2 * costh2 * costh2) * np.exp(-tanth2/roughness2)
       # Shadowing masking term for Beckmann: eq 27
       costhetav = ABS(VdotN)
       tanthetav = SQRT(1 - costhetav*costhetav)/costhetav
       a = 1.0/(roughness * tanthetav)

       iG1i = np.where(a < 1.6, BeckmannG1(VdotH/VdotN, a), BeckmannG2(VdotH/VdotN))

       costhetal = ABS(LdotN)
       tanthetal = SQRT(1 - costhetal*costhetal)/costhetal

       a = 1.0/(roughness * tanthetal)
       iG1o = np.where(a < 1.6, BeckmannG1(LdotH/LdotN, a), BeckmannG2(LdotH/LdotN))

       G = iG1i * iG1o

    else:
        raise ValueError('dist is neither G nor B')

    # Final BRDF value and PDF: eq 41
    # Refraction case
    denom = (VdotH + eta_o/eta_i * LdotH)**2
    idenom = 1.0 / denom
    fJacobian = absLdotH * idenom
    rJacobian = absVdotH * idenom

    # side check
    correct_refract = wi[...,2] * wo[...,2] <0
    value = np.where(correct_refract[nax], tAlbedo * ( (1-F) * D * G * absVdotH * fJacobian * (eta_o/eta_i)**2 / (ABS( VdotN ) *ABS( LdotN )))[nax], makeRtColorRGB(0.0)) # not baking LdotN
    fpdf = np.where(correct_refract, D * costheta * fJacobian * (eta_o/eta_i)**2, 0)
    rpdf = np.where(correct_refract, D * costheta * rJacobian, 0)

    return (wo, value, fpdf, rpdf)
