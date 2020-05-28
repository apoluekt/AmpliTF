# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import operator
import tensorflow as tf
import numpy as np
import math
import itertools

import amplitf.interface as atfi


@atfi.function
def spatial_components(vector):
    """Return spatial components of the input Lorentz vector

    :param vector: input Lorentz vector
    :returns: tensor of spatial components

    """
    return vector[..., 0:3]


@atfi.function
def time_component(vector):
    """Return time component of the input Lorentz vector

    :param vector: input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    :returns: vector of time components

    """
    return vector[..., 3]


@atfi.function
def x_component(vector):
    """Return spatial X component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of X-components

    """
    return vector[..., 0]



@atfi.function
def y_component(vector):
    """Return spatial Y component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Y-components

    """
    return vector[..., 1]



@atfi.function
def z_component(vector):
    """Return spatial Z component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Z-components

    """
    return vector[..., 2]



@atfi.function
def pt(vector):
    """Return transverse (X-Y) component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of transverse components

    """
    return atfi.sqrt(x_component(vector) ** 2 + y_component(vector) ** 2)



@atfi.function
def eta(vector):
    """Return pseudorapidity component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of pseudorapidity components

    """
    return -atfi.log(pt(vector) / 2. / z_component(vector))



@atfi.function
def vector(x, y, z):
    """Make a 3-vector from components
      x, y, z : vector components

    :param x: 
    :param y: 
    :param z: 

    """
    return tf.stack([x, y, z], axis=-1)



@atfi.function
def scalar(x):
    """Create a scalar (e.g. tensor with only one component) which can be used to e.g. scale a vector
    One cannot do e.g. const(2.)*vector(x, y, z), needs to do scalar(const(2))*vector(x, y, z)

    :param x: 

    """
    return tf.stack([x], axis=-1)



@atfi.function
def lorentz_vector(space, time):
    """Make a Lorentz vector from spatial and time components
      space : 3-vector of spatial components
      time  : time component

    :param space: 
    :param time: 

    """
    return tf.concat([space, tf.stack([time], axis=-1)], axis=-1)



@atfi.function
def metric_tensor():
    """Metric tensor for Lorentz space (constant)"""
    return tf.constant([-1., -1., -1., 1.], dtype=atfi.fptype())


@atfi.function
def mass_squared(vector):
    """Calculate mass scalar for Lorentz 4-momentum
      vector : input Lorentz momentum vector

    :param vector: 

    """
    return tf.reduce_sum(vector * vector * metric_tensor(), -1)



@atfi.function
def mass(vector):
    """Calculate mass scalar for Lorentz 4-momentum
      vector : input Lorentz momentum vector

    :param vector: 

    """
    return atfi.sqrt(mass_squared(vector))


@atfi.function
def scalar_product(vec1, vec2):
    """Calculate scalar product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.reduce_sum(vec1*vec2, -1)



@atfi.function
def vector_product(vec1, vec2):
    """Calculate vector product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.linalg.cross(vec1, vec2)



@atfi.function
def cross_product(vec1, vec2):
    """Calculate cross product of two 3-vectors

    :param vec1: 
    :param vec2: 

    """
    return tf.linalg.cross(vec1, vec2)



@atfi.function
def norm(vec):
    """Calculate norm of 3-vector

    :param vec: 

    """
    return atfi.sqrt(tf.reduce_sum(vec * vec, -1))



@atfi.function
def p(vector):
    """Calculate absolute value of the 4-momentum

    :param vector: 

    """
    return norm(spatial_components(vector))



@atfi.function
def unit_vector(vec):
    """Unit vector in the direction of vec

    :param vec: 

    """
    return vec / scalar(norm(vec))



@atfi.function
def perpendicular_unit_vector(vec1, vec2):
    """Unit vector perpendicular to the plane formed by vec1 and vec2

    :param vec1: 
    :param vec2: 

    """
    v = vector_product(vec1, vec2)
    return v / scalar(norm(v))



@atfi.function
def lorentz_boost(vector, boostvector):
    """Perform Lorentz boost
      vector :     4-vector to be boosted
      boostvector: boost vector. Can be either 3-vector or 4-vector (only spatial components are used)

    :param vector: 
    :param boostvector: 

    """
    boost = spatial_components(boostvector)
    b2 = scalar_product(boost, boost)
    gamma = 1./atfi.sqrt(1. - b2)
    gamma2 = (gamma-1.0)/b2
    ve = time_component(vector)
    vp = spatial_components(vector)
    bp = scalar_product(vp, boost)
    vp2 = vp + scalar(gamma2 * bp + gamma * ve) * boost
    ve2 = gamma*(ve + bp)
    return lorentz_vector(vp2, ve2)



@atfi.function
def boost_to_rest(vector, boostvector):
    """Perform Lorentz boost to the rest frame of the 4-vector boostvector.

    :param vector: 
    :param boostvector: 

    """
    boost = -spatial_components(boostvector) / scalar(time_component(boostvector))
    return lorentz_boost(vector, boost)



@atfi.function
def boost_from_rest(vector, boostvector):
    """Perform Lorentz boost from the rest frame of the 4-vector boostvector.

    :param vector: 
    :param boostvector: 

    """
    boost = spatial_components(boostvector) / scalar(time_component(boostvector))
    return lorentz_boost(vector, boost)



@atfi.function
def rotate(v, angle, axis):
    """rotate vector around an arbitrary axis, from ROOT implementation

    :param v: 
    :param angle: 
    :param axis: 

    """
    if (angle != atfi.zeros(angle)):
        ll = norm(axis)
        if (ll == atfi.zeros(ll)):
            sys.exit('ERROR in rotate: rotation axis is zero')
        else:
            sa = atfi.sin(angle)
            ca = atfi.cos(angle)
            dx = x_component(axis) / ll
            dy = y_component(axis) / ll
            dz = z_component(axis) / ll
            vx = x_component(v)
            vy = y_component(v)
            vz = z_component(v)
            _vx = (ca+(1-ca)*dx*dx)*vx + ((1-ca)*dx*dy-sa*dz) * \
                vy + ((1-ca)*dx*dz+sa*dy)*vz
            _vy = ((1-ca)*dy*dx+sa*dz)*vx + (ca+(1-ca)*dy*dy) * \
                vy + ((1-ca)*dy*dz-sa*dx)*vz
            _vz = ((1-ca)*dz*dx-sa*dy)*vx + \
                ((1-ca)*dz*dy+sa*dx)*vy + (ca+(1-ca)*dz*dz)*vz

            return vector(_vx, _vy, _vz)

    else:
        return v



@atfi.function
def rotate_euler(v, phi, theta, psi):
    """Perform 3D rotation of the 3-vector
      v : vector to be rotated
      phi, theta, psi : Euler angles in Z-Y-Z convention

    :param v: 
    :param phi: 
    :param theta: 
    :param psi: 

    """

    # rotate Z (phi)
    c1 = atfi.cos(phi)
    s1 = atfi.sin(phi)
    c2 = atfi.cos(theta)
    s2 = atfi.sin(theta)
    c3 = atfi.cos(psi)
    s3 = atfi.sin(psi)

    # rotate Y (theta)
    fzx2 = -s2*c1
    fzy2 = s2*s1
    fzz2 = c2

    # rotate Z (psi)
    fxx3 = c3*c2*c1 - s3*s1
    fxy3 = -c3*c2*s1 - s3*c1
    fxz3 = c3*s2
    fyx3 = s3*c2*c1 + c3*s1
    fyy3 = -s3*c2*s1 + c3*c1
    fyz3 = s3*s2

    # Transform v
    vx = x_component(v)
    vy = y_component(v)
    vz = z_component(v)

    _vx = fxx3*vx + fxy3*vy + fxz3*vz
    _vy = fyx3*vx + fyy3*vy + fyz3*vz
    _vz = fzx2*vx + fzy2*vy + fzz2*vz

    return vector(_vx, _vy, _vz)



@atfi.function
def rotate_lorentz_vector(v, phi, theta, psi):
    """Perform 3D rotation of the 4-vector
      v : vector to be rotated
      phi, theta, psi : Euler angles in Z-Y-Z convention

    :param v: 
    :param phi: 
    :param theta: 
    :param psi: 

    """
    return lorentz_vector(rotate_euler(spatial_components(v), phi, theta, psi), time_component(v))



@atfi.function
def project_lorentz_vector(p, axes):
    """

    :param p: 
    :param axes: 

    """
    (x1, y1, z1) = axes
    p0 = spatial_components(p)
    p1 = lorentz_vector(vector(scalar_product(x1, p0), scalar_product(
        y1, p0), scalar_product(z1, p0)), time_component(p))
    return p1



@atfi.function
def cos_helicity_angle_dalitz(m2ab, m2bc, md, ma, mb, mc):
    """Calculate cos(helicity angle) for set of two Dalitz plot variables
      m2ab, m2bc : Dalitz plot variables (inv. masses squared of AB and BC combinations)
      md : mass of the decaying particle
      ma, mb, mc : masses of final state particles

    :param m2ab: 
    :param m2bc: 
    :param md: 
    :param ma: 
    :param mb: 
    :param mc: 

    """
    md2 = md**2
    ma2 = ma**2
    mb2 = mb**2
    mc2 = mc**2
    m2ac = md2 + ma2 + mb2 + mc2 - m2ab - m2bc
    mab = atfi.sqrt(m2ab)
    mac = atfi.sqrt(m2ac)
    mbc = atfi.sqrt(m2bc)
    p2a = 0.25/md2*(md2-(mbc+ma)**2)*(md2-(mbc-ma)**2)
    p2b = 0.25/md2*(md2-(mac+mb)**2)*(md2-(mac-mb)**2)
    p2c = 0.25/md2*(md2-(mab+mc)**2)*(md2-(mab-mc)**2)
    eb = (m2ab-ma2+mb2)/2./mab
    ec = (md2-m2ab-mc2)/2./mab
    pb = atfi.sqrt(eb ** 2 - mb2)
    pc = atfi.sqrt(ec ** 2 - mc2)
    e2sum = (eb+ec)**2
    m2bc_max = e2sum-(pb-pc)**2
    m2bc_min = e2sum-(pb+pc)**2
    return (m2bc_max + m2bc_min - 2.*m2bc)/(m2bc_max-m2bc_min)



@atfi.function
def spherical_angles(pb):
    """theta, phi : polar and azimuthal angles of the vector pb

    :param pb: 

    """
    z1 = unit_vector(spatial_components(pb))       # New z-axis is in the direction of pb
    theta = atfi.acos(z_component(z1))                 # Helicity angle
    phi = atfi.atan2(y_component(pb), x_component(pb))  # phi angle
    return (theta, phi)



@atfi.function
def helicity_angles(pb):
    """theta, phi : polar and azimuthal angles of the vector pb

    :param pb: 

    """
    return spherical_angles(pb)



@atfi.function
def four_momenta_from_helicity_angles(md, ma, mb, theta, phi):
    """Calculate the four-momenta of the decay products in D->AB in the rest frame of D
        md:    mass of D
        ma:    mass of A
        mb:    mass of B
        theta: angle between A momentum in D rest frame and D momentum in its helicity frame
        phi:   angle of plane formed by A & B in D helicity frame

    :param md: 
    :param ma: 
    :param mb: 
    :param theta: 
    :param phi: 

    """
    # Calculate magnitude of momentum in D rest frame
    p = two_body_momentum(md, ma, mb)
    # Calculate energy in D rest frame
    Ea = atfi.sqrt(p ** 2 + ma ** 2)
    Eb = atfi.sqrt(p ** 2 + mb ** 2)
    # Construct four-momenta with A aligned with D in D helicity frame
    Pa = lorentz_vector(vector(atfi.zeros(p), atfi.zeros(p), p), Ea)
    Pb = lorentz_vector(vector(atfi.zeros(p), atfi.zeros(p), -p), Eb)
    # rotate four-momenta
    Pa = rotate_lorentz_vector(Pa, atfi.zeros(phi), -theta, -phi)
    Pb = rotate_lorentz_vector(Pb, atfi.zeros(phi), -theta, -phi)
    return Pa, Pb



#@atfi.function
def recursive_sum(vectors):
    """Helper function fro nested_helicity_angles. It sums all the vectors in
      a list or nested list

    :param vectors: 

    """
    return sum([recursive_sum(vector) if isinstance(vector, list) else vector for vector in vectors])



#@atfi.function
def nested_helicity_angles(pdecays):
    """Calculate the Helicity Angles for every decay topology specified with brackets []
    examples:
       - input:
         A -> B (-> C D) E (-> F G) ==> nested_helicity_angles([[C,D],[F,G]])
         A -> B (-> C (-> D E) F) G ==> nested_helicity_angles([ [ [ D, E] , F ] , G ])
       - output:
         A -> B (-> C D) E (-> F G) ==> (thetaB,phiB,thetaC,phiC,thetaF,phiF)
         A -> B (-> C (-> D E) F) G ==> (thetaB,phiB,thetaC,phiC,thetaD,phiD)
         where thetaX,phiX are the polar and azimuthal angles of X in the mother rest frame

    :param pdecays: 

    """
    angles = ()
    if len(pdecays) != 2:
        sys.exit(
            'ERROR in nested_helicity_angles: lenght of the input list is different from 2')

    for i, pdau in enumerate(pdecays):
        if i == 0:
            angles += helicity_angles(recursive_sum(pdau)
                                     if isinstance(pdau, list) else pdau)
        # the particle is not basic but decay, rotate and boost to its new rest frame
        if isinstance(pdau, list):
            pmother = recursive_sum(pdau)
            pdau_newframe = rotation_and_boost(pdau, pmother)
            angles += nested_helicity_angles(pdau_newframe)
    return angles



@atfi.function
def change_axes(ps, newaxes):
    """List of lorentz_vector with the component described by the
      new axes (x,y,z).

    :param ps: 
    :param newaxes: 

    """
    (xnew, ynew, znew) = newaxes
    pout = []
    for p in ps:
        px = x_component(p)
        py = y_component(p)
        pz = z_component(p)
        pout.append(lorentz_vector(vector(px * x_component(xnew) + py * y_component(xnew) + pz * z_component(xnew),
                                          px * x_component(ynew) + py * y_component(ynew) + pz * z_component(ynew),
                                          px * x_component(znew) + py * y_component(znew) + pz * z_component(znew)),
                    time_component(p)))
    return pout



@atfi.function
def axes_after_rotation(pb, oldaxes=None):
    """Calculate new (rotated) axes aligned with the momentum vector pb

    :param pb: 
    :param oldaxes:  (Default value = None)

    """
    z1 = unit_vector(spatial_components(pb))       # New z-axis is in the direction of pb
    eb = time_component(pb)
    zeros = atfi.zeros(eb)
    ones = atfi.ones(eb)
    # Old z-axis vector
    z0 = vector(zeros, zeros, ones) if oldaxes == None else oldaxes[2]
    # Old x-axis vector
    x0 = vector(ones, zeros, zeros) if oldaxes == None else oldaxes[0]
    sp = scalar_product(z1, z0)
    a0 = z0 - z1 * scalar(sp)   # vector in z-pb plane perpendicular to z0
    x1 = tf.where(scalar(tf.equal(sp, 1.)), x0, -unit_vector(a0))
    y1 = vector_product(z1, x1)                   # New y-axis
    return (x1, y1, z1)



@atfi.function
def axes_before_rotation(pb):
    """Calculate old (before rotation) axes in the frame aligned with the momentum vector pb

    :param pb: 

    """
    z1 = unit_vector(spatial_components(pb))       # New z-axis is in the direction of pb
    eb = time_component(pb)
    z0 = vector(atfi.zeros(eb), atfi.zeros(eb), atfi.ones(eb))  # Old z-axis vector
    x0 = vector(atfi.ones(eb), atfi.zeros(eb), atfi.zeros(eb))  # Old x-axis vector
    sp = scalar_product(z1, z0)
    a0 = z0 - z1 * scalar(sp)   # vector in z-pb plane perpendicular to z0
    x1 = tf.where(tf.equal(sp, 1.0), x0, -unit_vector(a0))
    y1 = vector_product(z1, x1)                   # New y-axis
    x = vector(x_component(x1), x_component(y1), x_component(z1))
    y = vector(y_component(x1), y_component(y1), y_component(z1))
    z = vector(z_component(x1), z_component(y1), z_component(z1))
    return (x, y, z)



@atfi.function
def rotation_and_boost(ps, pb):
    """rotate and boost all momenta from the list ps to the rest frame of pb
      After the rotation, the coordinate system is defined as:
        z axis: direction of pb
        y axis: perpendicular to the plane formed by the old z and pb
        x axis: [y,z]
    
      ps : list of Lorentz vectors to rotate and boost
      pb : Lorentz vector defining the new frame

    :param ps: 
    :param pb: 
    :returns: list of transformed Lorentz vectors
    :rtype: ps1

    """
    newaxes = axes_after_rotation(pb)
    eb = time_component(pb)
    zeros = atfi.zeros(eb)
    # Boost vector in the rotated coordinates along z axis
    boost = vector(zeros, zeros, -norm(spatial_components(pb)) / eb)

    return nested_rotation_and_boost(ps, newaxes, boost)



#@atfi.function
def nested_rotation_and_boost(ps, axes, boost):
    """Helper function for rotation_and_boost. It applies rotation_and_boost iteratively on nested lists

    :param ps: 
    :param axes: 
    :param boost: 

    """
    (x, y, z) = axes
    ps1 = []
    for p in ps:
        if isinstance(p, list):
            p2 = nested_rotation_and_boost(p, (x, y, z), boost)
        else:
            p1 = project_lorentz_vector(p, (x, y, z))
            p2 = lorentz_boost(p1, boost)
        ps1 += [p2]
    return ps1



@atfi.function
def euler_angles(x1, y1, z1, x2, y2, z2):
    """Calculate Euler angles (phi, theta, psi in the ZYZ convention) which transform the coordinate basis (x1, y1, z1)
      to the basis (x2, y2, z2). Both x1,y1,z1 and x2,y2,z2 are assumed to be orthonormal and right-handed.

    :param x1: 
    :param y1: 
    :param z1: 
    :param x2: 
    :param y2: 
    :param z2: 

    """
    theta = atfi.acos(scalar_product(z1, z2))
    phi = atfi.atan2(scalar_product(z1, y2), scalar_product(z1, x2))
    psi = atfi.atan2(scalar_product(y1, z2), scalar_product(x1, z2))
    return (phi, theta, psi)



@atfi.function
def helicity_angles_3body(pa, pb, pc):
    """Calculate 4 helicity angles for the 3-body D->ABC decay defined as:
      theta_r, phi_r : polar and azimuthal angles of the AB resonance in the D rest frame
      theta_a, phi_a : polar and azimuthal angles of the A in AB rest frame

    :param pa: 
    :param pb: 
    :param pc: 

    """
    theta_r = atfi.acos(-z_component(pc) / norm(spatial_components(pc)))
    phi_r = atfi.atan2(-y_component(pc), -x_component(pc))

    pa_prime = lorentz_vector(rotate_euler(spatial_components(
        pa), -phi_r, atfi.pi() - theta_r, phi_r), time_component(pa))
    pb_prime = lorentz_vector(rotate_euler(spatial_components(
        pb), -phi_r, atfi.pi() - theta_r, phi_r), time_component(pb))

    w = time_component(pa) + time_component(pb)

    pab = lorentz_vector(-(pa_prime + pb_prime) / scalar(w), w)
    pa_prime2 = lorentz_boost(pa_prime, pab)

    theta_a = atfi.acos(z_component(pa_prime2) / norm(spatial_components(pa_prime2)))
    phi_a = atfi.atan2(y_component(pa_prime2), x_component(pa_prime2))

    return (theta_r, phi_r, theta_a, phi_a)



@atfi.function
def cos_helicity_angle(p1, p2):
    """The helicity angle is defined as the angle between one of the two momenta in the p1+p2 rest frame
      with respect to the momentum of the p1+p2 system in the decaying particle rest frame (ptot)

    :param p1: 
    :param p2: 

    """
    p12 = lorentz_vector(spatial_components(p1) + spatial_components(p2),
                         time_component(p1) + time_component(p2))
    pcm1 = boost_to_rest(p1, p12)
    cosHel = scalar_product(unit_vector(spatial_components(pcm1)),
                            unit_vector(spatial_components(p12)))
    return cosHel



@atfi.function
def azimuthal_4body_angle(p1, p2, p3, p4):
    """Calculates the angle between the plane defined by (p1,p2) and (p3,p4)

    :param p1: 
    :param p2: 
    :param p3: 
    :param p4: 

    """
    v1 = spatial_components(p1)
    v2 = spatial_components(p2)
    v3 = spatial_components(p3)
    v4 = spatial_components(p4)
    n12 = unit_vector(vector_product(v1, v2))
    n34 = unit_vector(vector_product(v3, v4))
    z = unit_vector(v1 + v2)
    cosPhi = scalar_product(n12, n34)
    sinPhi = scalar_product(vector_product(n12, n34), z)
    phi = atfi.atan2(sinPhi, cosPhi)  # defined in [-pi,pi]
    return phi



@atfi.function
def helicity_angles_4body(pa, pb, pc, pd):
    """Calculate 4 helicity angles for the 4-body E->ABCD decay defined as:
      theta_ab, phi_ab : polar and azimuthal angles of the AB resonance in the E rest frame
      theta_cd, phi_cd : polar and azimuthal angles of the CD resonance in the E rest frame
      theta_ac, phi_ac : polar and azimuthal angles of the AC resonance in the E rest frame
      theta_bd, phi_bd : polar and azimuthal angles of the BD resonance in the E rest frame
      theta_ad, phi_ad : polar and azimuthal angles of the AD resonance in the E rest frame
      theta_bc, phi_bc : polar and azimuthal angles of the BC resonance in the E rest frame
      phi_ab_cd : azimuthal angle between AB and CD
      phi_ac_bd : azimuthal angle between AC and BD
      phi_ad_bc : azimuthal angle between AD and BC

    :param pa: 
    :param pb: 
    :param pc: 
    :param pd: 

    """
    theta_r = atfi.acos(-z_component(pc) / norm(spatial_components(pc)))
    phi_r = atfi.atan2(-y_component(pc), -x_component(pc))

    pa_prime = lorentz_vector(rotate_euler(spatial_components(pa),
                                           -phi_r, atfi.pi() - theta_r, phi_r), time_component(pa))
    pb_prime = lorentz_vector(rotate_euler(spatial_components(pb),
                                           -phi_r, atfi.pi() - theta_r, phi_r), time_component(pb))

    w = time_component(pa) + time_component(pb)

    pab = lorentz_vector(-(pa_prime + pb_prime) / scalar(w), w)
    pa_prime2 = lorentz_boost(pa_prime, pab)

    theta_a = atfi.acos(z_component(pa_prime2) / norm(spatial_components(pa_prime2)))
    phi_a = atfi.atan2(y_component(pa_prime2), x_component(pa_prime2))

    return (theta_r, phi_r, theta_a, phi_a)



@atfi.function
def wigner_capital_d(phi, theta, psi, j, m1, m2):
    """Calculate Wigner capital-D function.
      phi,
      theta,
      psi  : Rotation angles
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2, e.g. 1 for projection 1/2)

    :param phi: 
    :param theta: 
    :param psi: 
    :param j2: 
    :param m2_1: 
    :param m2_2: 

    """
    i = atfi.complex(atfi.const(0), atfi.const(1))
    return atfi.exp(-i*atfi.cast_complex(m1/2.*phi)) * \
           atfi.cast_complex(wigner_small_d(theta, j, m1, m2)) * \
           atfi.exp(-i * atfi.cast_complex(m2 / 2. * psi))


@atfi.function
def wigner_small_d(theta, j, m1, m2):
    """Calculate Wigner small-d function. Needs sympy.
      theta : angle
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2)

    :param theta: 
    :param j: 
    :param m1: 
    :param m2: 

    """
    from sympy import Rational
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify
    from sympy.physics.quantum.spin import Rotation as Wigner
    d = Wigner.d(Rational(j, 2), Rational(m1, 2),
                 Rational(m2, 2), x).doit().evalf()
    return lambdify(x, d, "tensorflow")(theta)


@atfi.function
def legendre(n, var):
    """Calculate Legendre_n(var)
      var : angle

    :param n: 
    :param var: 

    """
    from sympy import Rational
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify
    from sympy import legendre
    l = legendre(Rational(n), x)
    return lambdify(x, l, "tensorflow")(var)



@atfi.function
def spin_rotation_angle(pa, pb, pc, bachelor=2):
    """Calculate the angle between two spin-quantisation axes for the 3-body D->ABC decay
      aligned along the particle B and particle A.
        pa, pb, pc : 4-momenta of the final-state particles
        bachelor : index of the "bachelor" particle (0=A, 1=B, or 2=C)

    :param pa: 
    :param pb: 
    :param pc: 
    :param bachelor:  (Default value = 2)

    """
    if bachelor == 2:
        return atfi.const(0.)
    pboost = lorentz_vector(-spatial_components(pb) /
                            scalar(time_component(pb)), time_component(pb))
    if bachelor == 0:
        pa1 = spatial_components(lorentz_boost(pa, pboost))
        pc1 = spatial_components(lorentz_boost(pc, pboost))
        return atfi.acos(scalar_product(pa1, pc1) / norm(pa1) / norm(pc1))
    if bachelor == 1:
        pac = pa + pc
        pac1 = spatial_components(lorentz_boost(pac, pboost))
        pa1 = spatial_components(lorentz_boost(pa, pboost))
        return atfi.acos(scalar_product(pac1, pa1) / norm(pac1) / norm(pa1))
    return None



@atfi.function
def helicity_amplitude_3body(thetaR, phiR, thetaA, phiA, spinD, spinR, mu, lambdaR, lambdaA, lambdaB, lambdaC):
    """Calculate complex helicity amplitude for the 3-body decay D->ABC
      thetaR, phiR : polar and azimuthal angles of AB resonance in D rest frame
      thetaA, phiA : polar and azimuthal angles of A in AB rest frame
      spinD : D spin
      spinR : spin of the intermediate R resonance
      mu : D spin projection onto z axis
      lambdaR : R resonance helicity
      lambdaA : A helicity
      lambdaB : B helicity
      lambdaC : C helicity

    :param thetaR: 
    :param phiR: 
    :param thetaA: 
    :param phiA: 
    :param spinD: 
    :param spinR: 
    :param mu: 
    :param lambdaR: 
    :param lambdaA: 
    :param lambdaB: 
    :param lambdaC: 

    """

    lambda1 = lambdaR - lambdaC
    lambda2 = lambdaA - lambdaB
    ph = (mu-lambda1)/2.*phiR + (lambdaR-lambda2)/2.*phiA
    d_terms = wigner_small_d(thetaR, spinD, mu, lambda1) * \
              wigner_small_d(thetaA, spinR, lambdaR, lambda2)
    h = atfi.complex(d_terms * atfi.cos(ph), d_terms * atfi.sin(ph))

    return h



@atfi.function
def helicity_couplings_from_ls(ja, jb, jc, lb, lc, bls):
    """Helicity couplings from a list of LS couplings.
        ja : spin of A (decaying) particle
        jb : spin of B (1st decay product)
        jc : spin of C (2nd decay product)
        lb : B helicity
        lc : C helicity
        bls : dictionary of LS couplings, where:
          keys are tuples corresponding to (L,S) pairs
          values are values of LS couplings
      Note that ALL j,l,s should be doubled, e.g. S=1 for spin-1/2, L=2 for p-wave etc.

    :param ja: 
    :param jb: 
    :param jc: 
    :param lb: 
    :param lc: 
    :param bls: 

    """
    a = 0.
    # print("%d %d %d %d %d" % (ja, jb, jc, lb, lc))
    for ls, b in bls.items():
        l = ls[0]
        s = ls[1]
        coeff = math.sqrt((l+1)/(ja+1))*atfi.clebsch(jb, lb, jc,   -lc,  s, lb-lc)*\
                                        atfi.clebsch( l,  0, s,  lb-lc, ja, lb-lc)

        if coeff : a += atfi.complex(atfi.const(float(coeff)), atfi.const(0.)) * b
    return a



@atfi.function
def zemach_tensor(m2ab, m2ac, m2bc, m2d, m2a, m2b, m2c, spin):
    """Zemach tensor for 3-body D->ABC decay

    :param m2ab: 
    :param m2ac: 
    :param m2bc: 
    :param m2d: 
    :param m2a: 
    :param m2b: 
    :param m2c: 
    :param spin: 

    """
    z = None
    if spin == 0:
        z = atfi.complex(atfi.const(1.), atfi.const(0.))
    if spin == 1:
        z = atfi.complex(m2ac - m2bc + (m2d - m2c) * (m2b - m2a) / m2ab, atfi.const(0.))
    if spin == 2:
        z = atfi.complex((m2bc - m2ac + (m2d - m2c) * (m2a - m2b) / m2ab) ** 2 - 1. / 3. * (m2ab - 2. * (m2d + m2c) +
                         (m2d-m2c) ** 2 / m2ab) * (m2ab-2.*(m2a+m2b)+(m2a-m2b)**2/m2ab), atfi.const(0.))
    return z



@atfi.function
def two_body_momentum(md, ma, mb):
    """Momentum of two-body decay products D->AB in the D rest frame

    :param md: 
    :param ma: 
    :param mb: 

    """
    return atfi.sqrt((md ** 2 - (ma + mb) ** 2) * (md ** 2 - (ma - mb) ** 2) / (4 * md ** 2))


@atfi.function
def complex_two_body_momentum(md, ma, mb):
    """Momentum of two-body decay products D->AB in the D rest frame.
      Output value is a complex number, analytic continuation for the
      region below threshold.

    :param md: 
    :param ma: 
    :param mb: 

    """
    return atfi.sqrt(atfi.complex((md ** 2 - (ma + mb) ** 2) * (md ** 2 - (ma - mb) ** 2) / (4 * md ** 2), atfi.const(0.)))
