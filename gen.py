# Copyright (c) 2007, Matt Heinzen
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT HEINZEN ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MATT HEINZEN BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys, os, random, time, math
from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import ode

ii = 0
def sign(x):
    """Returns 1.0 if x is positive, -1.0 if x is negative or zero."""
    if x > 0.0: return 1.0
    else: return -1.0

def len3(v):
    """Returns the length of 3-vector v."""
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def neg3(v):
    """Returns the negation of 3-vector v."""
    return (-v[0], -v[1], -v[2])

def add3(a, b):
    """Returns the sum of 3-vectors a and b."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def sub3(a, b):
    """Returns the difference between 3-vectors a and b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def mul3(v, s):
    """Returns 3-vector v multiplied by scalar s."""
    return (v[0] * s, v[1] * s, v[2] * s)

def div3(v, s):
    """Returns 3-vector v divided by scalar s."""
    return (v[0] / s, v[1] / s, v[2] / s)

def dist3(a, b):
    """Returns the distance between point 3-vectors a and b."""
    return len3(sub3(a, b))

def norm3(v):
    """Returns the unit length 3-vector parallel to 3-vector v."""
    l = len3(v)
    if (l > 0.0): return (v[0] / l, v[1] / l, v[2] / l)
    else: return (0.0, 0.0, 0.0)

def dot3(a, b):
    """Returns the dot product of 3-vectors a and b."""
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def cross(a, b):
    """Returns the cross product of 3-vectors a and b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0])

def project3(v, d):
    """Returns projection of 3-vector v onto unit 3-vector d."""
    return mul3(v, dot3(norm3(v), d))

def acosdot3(a, b):
    """Returns the angle between unit 3-vectors a and b."""
    x = dot3(a, b)
    if x < -1.0: return pi
    elif x > 1.0: return 0.0
    else: return acos(x)

def rotate3(m, v):
    """Returns the rotation of 3-vector v by 3x3 (row major) matrix m."""
    return (v[0] * m[0] + v[1] * m[1] + v[2] * m[2],
        v[0] * m[3] + v[1] * m[4] + v[2] * m[5],
        v[0] * m[6] + v[1] * m[7] + v[2] * m[8])

def invert3x3(m):
    """Returns the inversion (transpose) of 3x3 rotation matrix m."""
    return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])

def zaxis(m):
    """Returns the z-axis vector from 3x3 (row major) rotation matrix m."""
    return (m[2], m[5], m[8])

def calcRotMatrix(axis, angle):
    """
    Returns the row-major 3x3 rotation matrix defining a rotation around axis by
    angle.
    """
    cosTheta = cos(angle)
    sinTheta = sin(angle)
    t = 1.0 - cosTheta
    return (
        t * axis[0]**2 + cosTheta,
        t * axis[0] * axis[1] - sinTheta * axis[2],
        t * axis[0] * axis[2] + sinTheta * axis[1],
        t * axis[0] * axis[1] + sinTheta * axis[2],
        t * axis[1]**2 + cosTheta,
        t * axis[1] * axis[2] - sinTheta * axis[0],
        t * axis[0] * axis[2] - sinTheta * axis[1],
        t * axis[1] * axis[2] + sinTheta * axis[0],
        t * axis[2]**2 + cosTheta)

def makeOpenGLMatrix(r, p):
    """
    Returns an OpenGL compatible (column-major, 4x4 homogeneous) transformation
    matrix from ODE compatible (row-major, 3x3) rotation matrix r and position
    vector p.
    """
    return (
        r[0], r[3], r[6], 0.0,
        r[1], r[4], r[7], 0.0,
        r[2], r[5], r[8], 0.0,
        p[0], p[1], p[2], 1.0)

def getBodyRelVec(b, v):
    """
    Returns the 3-vector v transformed into the local coordinate system of ODE
    body b.
    """
    return rotate3(invert3x3(b.getRotation()), v)


# rotation directions are named by the third (z-axis) row of the 3x3 matrix,
#   because ODE capsules are oriented along the z-axis
rightRot = (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
leftRot = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
upRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
downRot = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
bkwdRot = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# axes used to determine constrained joint rotations
rightAxis = (1.0, 0.0, 0.0)
leftAxis = (-1.0, 0.0, 0.0)
upAxis = (0.0, 1.0, 0.0)
downAxis = (0.0, -1.0, 0.0)
bkwdAxis = (0.0, 0.0, 1.0)
fwdAxis = (0.0, 0.0, -1.0)

UPPER_ARM_LEN = 0.30
FORE_ARM_LEN = 0.25
HAND_LEN = 0.13 # wrist to mid-fingers only
FOOT_LEN = 0.18 # ankles to base of ball of foot only
HEEL_LEN = 0.05

BROW_H = 1.68
MOUTH_H = 1.53
NECK_H = 1.50
SHOULDER_H = 1.37
CHEST_H = 1.35
HIP_H = 0.86
KNEE_H = 0.48
ANKLE_H = 0.08

SHOULDER_W = 0.41
CHEST_W = 0.36 # actually wider, but we want narrower than shoulders (esp. with large radius)
LEG_W = 0.28 # between middles of upper legs
PELVIS_W = 0.25 # actually wider, but we want smaller than hip width

R_SHOULDER_POS = (-SHOULDER_W * 0.5, SHOULDER_H, 0.0)
L_SHOULDER_POS = (SHOULDER_W * 0.5, SHOULDER_H, 0.0)
R_ELBOW_POS = sub3(R_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
L_ELBOW_POS = add3(L_SHOULDER_POS, (UPPER_ARM_LEN, 0.0, 0.0))
R_WRIST_POS = sub3(R_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
L_WRIST_POS = add3(L_ELBOW_POS, (FORE_ARM_LEN, 0.0, 0.0))
R_FINGERS_POS = sub3(R_WRIST_POS, (HAND_LEN, 0.0, 0.0))
L_FINGERS_POS = add3(L_WRIST_POS, (HAND_LEN, 0.0, 0.0))

R_HIP_POS = (-LEG_W * 0.5, HIP_H, 0.0)
L_HIP_POS = (LEG_W * 0.5, HIP_H, 0.0)
R_KNEE_POS = (-LEG_W * 0.5, KNEE_H, 0.0)
L_KNEE_POS = (LEG_W * 0.5, KNEE_H, 0.0)
R_ANKLE_POS = (-LEG_W * 0.5, ANKLE_H, 0.0)
L_ANKLE_POS = (LEG_W * 0.5, ANKLE_H, 0.0)
R_HEEL_POS = sub3(R_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
L_HEEL_POS = sub3(L_ANKLE_POS, (0.0, 0.0, HEEL_LEN))
R_TOES_POS = add3(R_ANKLE_POS, (0.0, 0.0, FOOT_LEN))
L_TOES_POS = add3(L_ANKLE_POS, (0.0, 0.0, FOOT_LEN))

class RagDoll():
    def __init__(self, world, space, density, offset = (0.0, 0.0, 0.0)):
        """Creates a ragdoll of standard size at the given offset."""

        self.world = world
        self.space = space
        self.density = density
        self.bodies = []
        self.geoms = []
        self.joints = []

        self.totalMass = 0.0
        self.walking = 0
        self.walk_time_steps = 100
        self.walking_force = 60
        self.walk_time_counter = 0

        self.sitting = 0
        self.sit_state = 0
        self.sit_time_steps = 100
        self.sit_time_counter = 0

        self.punching = 0
        self.punch_state = 1
        self.punch_time_steps = 100
        self.punch_time_counter = 0

        self.kicking = 0
        self.kick_state = 1
        self.kick_time_steps = 100
        self.kick_time_counter = 0

        self.handshaking = 0
        self.handshake_state = 1
        self.handshake_time_steps = 100
        self.handshake_time_counter = 0
                
        self.jumping = 0
        self.jump_state = 1
        self.jump_time_steps = 100
        self.jump_time_counter = 0

        self.namaste_on = 0
        self.namaste_state = 1
        self.namaste_time_steps = 100
        self.namaste_time_counter = 0
        
        self.handwaving = 0
        self.handwave_state = 1
        self.handwave_time_steps = 100
        self.handwave_time_counter = 0

        self.offset = offset
        self.restoring_torque = 40

        skin = (0.8,0.6,0.3)
        shirt = (1,0,0)
        pants = (0.8,0.8,0.8)

        self.chest = self.addBody((-CHEST_W * 0.5, CHEST_H, 0.0),
            (CHEST_W * 0.5, CHEST_H, 0.0), 0.13, shirt)
        self.belly = self.addBody((0.0, CHEST_H - 0.1, 0.0),
            (0.0, HIP_H + 0.1, 0.0), 0.125, shirt)
        self.midSpine = self.addFixedJoint(self.chest, self.belly)
        self.pelvis = self.addBody((-PELVIS_W * 0.5, HIP_H, 0.0,(0,0,1)),
            (PELVIS_W * 0.5, HIP_H, 0.0), 0.125,pants)
        self.lowSpine = self.addFixedJoint(self.belly, self.pelvis)

        self.head = self.addBody((0.0, BROW_H, 0.0), (0.0, MOUTH_H, 0.0), 0.11,
                skin)
        #self.neck = self.addBallJoint(self.chest, self.head,
            #(0.0, NECK_H, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), pi * 0.25,
            #pi * 0.25, 80.0, 40.0)
        self.neck  = self.addFixedJoint(self.chest, self.head)

        self.rightUpperLeg = self.addBody(R_HIP_POS, R_KNEE_POS, 0.11,pants)
        self.rightHip = self.addUniversalJoint(self.pelvis, self.rightUpperLeg,
            R_HIP_POS, bkwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
            0.75 * pi)
        self.leftUpperLeg = self.addBody(L_HIP_POS, L_KNEE_POS, 0.11,pants)
        self.leftHip = self.addUniversalJoint(self.pelvis, self.leftUpperLeg,
            L_HIP_POS, fwdAxis, rightAxis, -0.1 * pi, 0.3 * pi, -0.15 * pi,
            0.75 * pi)

        self.rightLowerLeg = self.addBody(R_KNEE_POS, R_ANKLE_POS, 0.09,skin)
        self.rightKnee = self.addHingeJoint(self.rightUpperLeg,
            self.rightLowerLeg, R_KNEE_POS, leftAxis, 0.0, pi * 0.75)
        self.leftLowerLeg = self.addBody(L_KNEE_POS, L_ANKLE_POS, 0.09,skin)
        self.leftKnee = self.addHingeJoint(self.leftUpperLeg,
            self.leftLowerLeg, L_KNEE_POS, leftAxis, 0.0, pi * 0.75)

        #self.rightFoot = self.addBody(R_HEEL_POS, R_TOES_POS, 0.09)
        #self.rightAnkle = self.addHingeJoint(self.rightLowerLeg,
            #self.rightFoot, R_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)
        #self.rightAnkle = self.addFixedJoint(self.rightLowerLeg,self.rightFoot)
        #self.leftFoot = self.addBody(L_HEEL_POS, L_TOES_POS, 0.09)
        #self.leftAnkle = self.addFixedJoint(self.leftLowerLeg,self.leftFoot)
        #self.leftAnkle = self.addHingeJoint(self.leftLowerLeg,
            #self.leftFoot, L_ANKLE_POS, rightAxis, -0.1 * pi, 0.05 * pi)

        self.rightUpperArm = self.addBody(R_SHOULDER_POS, R_ELBOW_POS,
                0.08,shirt)
        self.rightShoulder = self.addBallJoint(self.chest, self.rightUpperArm,
            R_SHOULDER_POS, norm3((-1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)
        self.leftUpperArm = self.addBody(L_SHOULDER_POS, L_ELBOW_POS,
                0.08,shirt)
        self.leftShoulder = self.addBallJoint(self.chest, self.leftUpperArm,
            L_SHOULDER_POS, norm3((1.0, -1.0, 4.0)), (0.0, 0.0, 1.0), pi * 0.5,
            pi * 0.25, 150.0, 100.0)

        self.rightForeArm = self.addBody(R_ELBOW_POS, R_WRIST_POS, 0.075,skin)
        self.rightElbow = self.addHingeJoint(self.rightUpperArm,
            self.rightForeArm, R_ELBOW_POS, downAxis, 0.0, 0.6 * pi)
        self.leftForeArm = self.addBody(L_ELBOW_POS, L_WRIST_POS, 0.075,skin)
        self.leftElbow = self.addHingeJoint(self.leftUpperArm,
            self.leftForeArm, L_ELBOW_POS, upAxis, 0.0, 0.6 * pi)

        self.rightHand = self.addBody(R_WRIST_POS, R_FINGERS_POS, 0.075,skin)
        self.rightHand.moving = False
        self.rightWrist = self.addHingeJoint(self.rightForeArm,
            self.rightHand, R_WRIST_POS, fwdAxis, -0.1 * pi, 0.2 * pi)
        self.leftHand = self.addBody(L_WRIST_POS, L_FINGERS_POS, 0.075,skin)
        self.leftWrist = self.addHingeJoint(self.leftForeArm,
            self.leftHand, L_WRIST_POS, bkwdAxis, -0.1 * pi, 0.2 * pi)

        self.belly.stabilize = True
        self.rightUpperLeg.stabilize = True
        self.rightLowerLeg.stabilize = True
        self.leftUpperLeg.stabilize = True
        self.leftLowerLeg.stabilize = True
        self.rightUpperArm.stabilize = True
        self.leftUpperArm.stabilize = True

        self.belly.lefttilt = True
        self.belly.righttilt = True

        self.belly.stabilizing_str = 150

        self.leftUpperLeg.stabilizing_str = 100
        self.leftLowerLeg.stabilizing_str = 100
        self.rightUpperLeg.stabilizing_str = 100
        self.rightLowerLeg.stabilizing_str = 100

        self.rightUpperArm.stabilizing_str = 10
        self.leftUpperArm.stabilizing_str = 10

        self.leftUpperLeg.tilt_str = 20
        self.leftLowerLeg.tilt_str = 30
        self.rightUpperLeg.tilt_str = 20
        self.rightLowerLeg.tilt_str = 30

        self.rightUpperArm.tilt_str = 20
        self.rightForeArm.tilt_str = 20
        self.leftUpperArm.tilt_str = 20
        self.leftForeArm.tilt_str = 20

        self.leftUpperLeg.tilt_time = 10
        self.leftLowerLeg.tilt_time = 10
        self.rightUpperLeg.tilt_time = 10
        self.rightLowerLeg.tilt_time = 10

        self.rightUpperArm.tilt_time = 1000
        self.rightForeArm.tilt_time = 1000
        self.leftUpperArm.tilt_time = 1000
        self.leftForeArm.tilt_time = 1000


    def addBody(self, p1, p2, radius, color=(0.8,0.8,0.8)):
        """
        Adds a capsule body between joint positions p1 and p2 and with given
        radius to the ragdoll.
        """

        p1 = add3(p1, self.offset)
        p2 = add3(p2, self.offset)

        # cylinder length not including endcaps, make capsules overlap by half
        #   radius at joints
        cyllen = dist3(p1, p2) - radius

        body = ode.Body(self.world)
        m = ode.Mass()
        m.setCylinder(self.density, 3, radius, cyllen)
        body.setMass(m)

        # set parameters for drawing the body
        body.shape = "capsule"
        body.length = cyllen
        body.radius = radius
        body.color = color

        # create a capsule geom for collision detection
        geom = ode.GeomCCylinder(self.space, radius, cyllen)
        geom.setBody(body)

        # define body rotation automatically from body axis
        za = norm3(sub3(p2, p1))
        if (abs(dot3(za, (1.0, 0.0, 0.0))) < 0.7): xa = (1.0, 0.0, 0.0)
        else: xa = (0.0, 1.0, 0.0)
        ya = cross(za, xa)
        xa = norm3(cross(ya, za))
        ya = cross(za, xa)
        rot = (xa[0], ya[0], za[0], xa[1], ya[1], za[1], xa[2], ya[2], za[2])

        body.setPosition(mul3(add3(p1, p2), 0.5))
        body.setRotation(rot)
        body.stabilize = False
        body.tilt = False
        self.bodies.append(body)
        self.geoms.append(geom)

        self.totalMass += body.getMass().mass

        return body

    def addFixedJoint(self, body1, body2):
        joint = ode.FixedJoint(self.world)
        joint.attach(body1, body2)
        joint.setFixed()

        joint.style = "fixed"
        self.joints.append(joint)

        return joint

    def addHingeJoint(self, body1, body2, anchor, axis, loStop = -ode.Infinity,
        hiStop = ode.Infinity):

        anchor = add3(anchor, self.offset)

        joint = ode.HingeJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis(axis)
        joint.setParam(ode.ParamLoStop, loStop)
        joint.setParam(ode.ParamHiStop, hiStop)

        joint.style = "hinge"
        self.joints.append(joint)

        return joint

    def addUniversalJoint(self, body1, body2, anchor, axis1, axis2,
        loStop1 = -ode.Infinity, hiStop1 = ode.Infinity,
        loStop2 = -ode.Infinity, hiStop2 = ode.Infinity):

        anchor = add3(anchor, self.offset)

        joint = ode.UniversalJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)
        joint.setAxis1(axis1)
        joint.setAxis2(axis2)
        joint.setParam(ode.ParamLoStop, loStop1)
        joint.setParam(ode.ParamHiStop, hiStop1)
        joint.setParam(ode.ParamLoStop2, loStop2)
        joint.setParam(ode.ParamHiStop2, hiStop2)

        joint.style = "univ"
        self.joints.append(joint)

        return joint

    def addBallJoint(self, body1, body2, anchor, baseAxis, baseTwistUp,
        flexLimit = pi, twistLimit = pi, flexForce = 0.0, twistForce = 0.0):

        anchor = add3(anchor, self.offset)

        # create the joint
        joint = ode.BallJoint(self.world)
        joint.attach(body1, body2)
        joint.setAnchor(anchor)

        # store the base orientation of the joint in the local coordinate system
        #   of the primary body (because baseAxis and baseTwistUp may not be
        #   orthogonal, the nearest vector to baseTwistUp but orthogonal to
        #   baseAxis is calculated and stored with the joint)
        joint.baseAxis = getBodyRelVec(body1, baseAxis)
        tempTwistUp = getBodyRelVec(body1, baseTwistUp)
        baseSide = norm3(cross(tempTwistUp, joint.baseAxis))
        joint.baseTwistUp = norm3(cross(joint.baseAxis, baseSide))

        # store the base twist up vector (original version) in the local
        #   coordinate system of the secondary body
        joint.baseTwistUp2 = getBodyRelVec(body2, baseTwistUp)

        # store joint rotation limits and resistive force factors
        joint.flexLimit = flexLimit
        joint.twistLimit = twistLimit
        joint.flexForce = flexForce
        joint.twistForce = twistForce

        joint.style = "ball"
        self.joints.append(joint)

        return joint

    def stabilise(self,body):
        """
        Set stablilizing_str for the body to customize strength
        """
        body_axis = rotate3(body.getRotation(), (0,0,1))
        torq_axis = cross((0,1,0),body_axis)
        ang = math.acos(body_axis[2])
        if body.stabilizing_str:
            body.addTorque(mul3(norm3(torq_axis),body.stabilizing_str*ang))
        else:
            body.addTorque(mul3(norm3(torq_axis),500*ang))


    def smooth_tilt(self,body):
        """
        Assumes that Final Tilt Direction for the body is set
        """
        body_axis = rotate3(body.getRotation(), (0,0,1))
        diff_axis = sub3(body.final_tilt_direction,body_axis)
        reduced_axis = add3(body_axis,div3(diff_axis,body.tilt_time))
        ang_vel = body.getAngularVel()
        torq_axis = cross(body_axis,reduced_axis )
        coherence = dot3(ang_vel,torq_axis)
        ang = math.acos(body_axis[2])
        if body.tilt_str:
            body.addTorque(mul3(norm3(torq_axis),body.tilt_str*ang))
        else:
            body.addTorque(mul3(norm3(torq_axis),500*ang))
        if coherence<0:
            #print coherence
            body.addTorque(mul3(norm3(torq_axis),ang*self.restoring_torque))


    def straighten(self,elbow):
        b1 = elbow.getBody(0)
        b2 = elbow.getBody(1)
        b1_axis = rotate3(b1.getRotation(), (0,0,1))
        b2_axis = rotate3(b2.getRotation(), (0,0,1))
        torq_axis = norm3(cross(b2_axis,b1_axis))
        ang = acosdot3(b1_axis, b2_axis)
        b1.addTorque(mul3(torq_axis, -10*ang))
        b2.addTorque(mul3(torq_axis, 10*ang))

    def moveto(self,bodypart,location):
        pos = bodypart.getPosition()
        diff = sub3(location,pos)
        if len3(diff) < 0.1:
            bodypart.moving = False
        else:
            bodypart.addForce(mul3(diff,200))

    def getUpAxis(self):
        ppos = self.pelvis.getPosition()
        bellypos = self.belly.getPosition()
        up_axis = norm3(sub3(bellypos,ppos))
        return up_axis

    def getRightAxis(self):
        ppos = self.pelvis.getPosition()
        rulpos = self.rightUpperLeg.getPosition()
        lulpos = self.leftUpperLeg.getPosition()
        lul_pel = sub3(ppos,lulpos)
        pel_rul = sub3(rulpos,ppos)
        right_axis = norm3(add3(lul_pel,pel_rul))
        return right_axis

    def getForwardAxis(self):
        right_axis = self.getRightAxis()
        up_axis = self.getUpAxis()
        front_axis = cross(up_axis,right_axis)
        return front_axis

    def sit(self):
        print "State - "+str(self.sit_state)
#        if self.sit_state==1:
#            self.initSitBack()
#            self.sit_state = 2
#            self.sit_time_steps = 400
        if self.sit_state==1:
            self.initSitBack()
            self.initSitFall()
            self.sit_state = 2
            self.sit_time_steps = 1200
        elif self.sit_state == 2:
            self.initSitStand()
            self.sit_state = 0
            self.sit_time_steps = 600
        elif self.sit_state == 0:
            self.sitReset()
            self.sitting = 0
            self.sit_time_steps = 300


    def walk(self):
        print "State - "+str(self.walk_state)
        if self.walk_state==1:
            self.initStandOnRightLeg()
            self.walk_state=2
            self.walk_time_steps = 150

        elif self.walk_state==2:
            self.initRestRightLeg()
            self.walk_state=3
            self.walk_time_steps = 150

        elif self.walk_state==3:
            self.initWalkLeftLegFront()
            self.walk_state=4
            self.walk_time_steps = 250

        elif self.walk_state==4:
            self.initStandOnLeftLeg()
            self.walk_state=5
            self.walk_time_steps = 150

        elif self.walk_state==5:
            self.initRestLeftLeg()
            self.walk_state=6
            self.walk_time_steps = 150

        elif self.walk_state==6:
            self.initWalkRightLegFront()
            self.walk_state=1
            self.walk_time_steps = 250


    def punch(self):
        print "State - "+str(self.punch_state)
        if self.punch_state==1:
            self.initPunchRaiseArm()
            self.punch_state=2
            self.punch_time_steps = 150

        elif self.punch_state==2:
            self.initPunchRaiseArm2()
            self.punch_state=3
            self.punch_time_steps = 150

        elif self.punch_state==3:
            self.initPunchExtendArm()
            self.punch_state = 4
            self.punch_time_steps = 50
        elif self.punch_state==4:
            self.finishPunch()

    def kick(self):
        print "State - "+str(self.kick_state)
        if self.kick_state==1:
            self.initKickLegBehind()
            self.kick_state=2
            self.kick_time_steps = 400
        elif self.kick_state==2:
            self.initKickLegFront()
            self.kick_state=3
            self.kick_time_steps = 400

        elif self.kick_state==3:
            self.finishKick()

    def handwave(self):
        print "State - "+str(self.handwave_state)
        if self.handwave_state==1:
            self.initHandWavePos1()
            self.handwave_state=2
            self.handwave_time_steps = 800
        elif self.handwave_state==2:
            self.initHandWavePos2()
            self.handwave_state=3
            self.handwave_time_steps = 250
        elif self.handwave_state==3:
            self.initHandWavePos3()
            self.handwave_state=2
            self.handwave_time_steps = 200

    def namaste(self):
        print "State - "+str(self.namaste_state)
        if self.namaste_state==1:
            self.initNamastePos1()
            self.namaste_state=2
            self.namaste_time_steps = 200
        elif self.namaste_state==2:
            self.initNamastePos2()
            self.namaste_state=3
            self.namaste_time_steps = 2500
        elif self.namaste_state==3:
            self.finishNamaste()

    def jump(self):
        print "State - "+str(self.jump_state)
        if self.jump_state==1:
            self.initPrepareForJump()
            self.jump_state=2
            self.jump_time_steps = 600
        elif self.jump_state==2:
            self.jump_state=3
            self.jump_time_steps = 25
        elif self.jump_state==3:
            self.finishJump()

    def handshake(self):
        print "State - "+str(self.handshake_state)
        if self.handshake_state==1:
            self.initHandshakePos1()
            self.handshake_state=2
            self.handshake_time_steps = 200
        elif self.handshake_state==2:
            self.handshake_state=1
            self.initHandshakePos2()
            self.handshake_time_steps = 200
            



    def move_hand_to_stable_pos(self):
        pass

    def kickass(self):
        self.pelvis.addForce(mul3((0,1,0),8000))

    def update(self):
        self.stabilise(self.belly)
        for b in self.bodies:
            if b.stabilize:
                self.stabilise(b)
            if b.tilt:
                self.smooth_tilt(b)

        if self.belly.lefttilt:
            self.belly.addTorque(mul3(self.getForwardAxis(),-100))
        if self.belly.righttilt:
            self.belly.addTorque(mul3(self.getForwardAxis(),100))

        self.stabilise(self.leftUpperArm)
        self.stabilise(self.rightUpperArm)
        self.straighten(self.leftElbow)
        self.straighten(self.rightElbow)

        if self.rightHand.moving:
            self.moveto(self.rightHand,self.rightHand.destination)

        if self.walking == 1:
            if self.walk_time_counter==self.walk_time_steps:
                self.walk()
                self.walk_time_counter = 0
            self.walk_time_counter+=1
            forward = self.getForwardAxis()
            self.bodies[1].addForce(mul3(forward,self.walking_force))

        if self.sitting == 1:
            if self.sit_time_counter == self.sit_time_steps:
                self.sit_time_counter = 0
                self.sit()
            self.sit_time_counter+=1

        if self.punching == 1:
            if self.punch_time_counter==self.punch_time_steps:
                self.punch()
                self.punch_time_counter = 0
            self.punch_time_counter+=1

        if self.handwaving == 1:
            if self.handwave_time_counter==self.handwave_time_steps:
                self.handwave()
                self.handwave_time_counter = 0
            self.handwave_time_counter+=1
        
        if self.jumping == 1:
            if self.jump_time_counter==self.jump_time_steps:
                self.jump_time_counter = 0
                self.jump()
            if self.jump_state==3:
                self.initJumpInAir()
            self.jump_time_counter+=1

        if self.handshaking == 1:
            if self.handshake_time_counter==self.handshake_time_steps:
                self.handshake_time_counter = 0
                self.handshake()
            self.handshake_time_counter+=1
        
        if self.namaste_on == 1:
            if self.namaste_time_counter==self.namaste_time_steps:
                self.namaste()
                self.namaste_time_counter = 0
            self.namaste_time_counter+=1

        if self.kicking == 1:
            if self.kick_time_counter==self.kick_time_steps:
                self.kick()
                self.kick_time_counter = 0
            self.kick_time_counter+=1
            self.pelvis.addTorque((0,-6,0))
        THRESH = 0.0
        ANG_THRESH = 0

        flexAngVel = self.bodies[1].getAngularVel();
        self.bodies[1].addTorque(mul3(flexAngVel,
            -0.5  ))
        for b in self.bodies:
            if len3(b.getLinearVel()) < THRESH:
                b.setLinearVel((0,0,0))
            if len3(b.getAngularVel()) < ANG_THRESH:
                b.setAngularVel((0,0,0))
        for j in self.joints:
            if j.style == "ball":
                # determine base and current attached body axes
                baseAxis = rotate3(j.getBody(0).getRotation(), j.baseAxis)
                currAxis = zaxis(j.getBody(1).getRotation())

                # get angular velocity of attached body relative to fixed body
                relAngVel = sub3(j.getBody(1).getAngularVel(),
                j.getBody(0).getAngularVel())
                twistAngVel = project3(relAngVel, currAxis)
                flexAngVel = sub3(relAngVel, twistAngVel)

                # restrict limbs rotating too far from base axis
                angle = acosdot3(currAxis, baseAxis)
                if angle > j.flexLimit:
                    # add torque to push body back towards base axis
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(currAxis, baseAxis)),
                        (angle - j.flexLimit) * j.flexForce))

                    # dampen flex to prevent bounceback
                    j.getBody(1).addTorque(mul3(flexAngVel,
                        -0.01 * j.flexForce))

                # determine the base twist up vector for the current attached
                #   body by applying the current joint flex to the fixed body's
                #   base twist up vector
                baseTwistUp = rotate3(j.getBody(0).getRotation(), j.baseTwistUp)
                base2current = calcRotMatrix(norm3(cross(baseAxis, currAxis)),
                    acosdot3(baseAxis, currAxis))
                projBaseTwistUp = rotate3(base2current, baseTwistUp)

                # determine the current twist up vector from the attached body
                actualTwistUp = rotate3(j.getBody(1).getRotation(),
                    j.baseTwistUp2)

                # restrict limbs twisting
                angle = acosdot3(actualTwistUp, projBaseTwistUp)
                if angle > j.twistLimit:
                    # add torque to rotate body back towards base angle
                    j.getBody(1).addTorque(mul3(
                        norm3(cross(actualTwistUp, projBaseTwistUp)),
                        (angle - j.twistLimit) * j.twistForce))

                    # dampen twisting
                    j.getBody(1).addTorque(mul3(twistAngVel,
                        -0.01 * j.twistForce))

    def getRelAxis(self,up_coeff,right_coeff,for_coeff):
        up = self.getUpAxis()
        right = self.getRightAxis()
        forward = self.getForwardAxis()
        axis = reduce(add3,[mul3(up,up_coeff),mul3(right,right_coeff),mul3(forward,for_coeff)])
        return axis

    def getRelPos(self,up_coeff,right_coeff,for_coeff):
        # Set Pelvis as origin
        ppos = self.pelvis.getPosition()
        up = self.getUpAxis()
        right = self.getRightAxis()
        forward = self.getForwardAxis()
        dest = reduce(add3,[mul3(up,up_coeff),mul3(right,right_coeff),mul3(forward,for_coeff),ppos])
        return dest



    def initSitBack(ragdoll):
        ragdoll.rightUpperLeg.final_tilt_direction =\
            add3(mul3(ragdoll.getForwardAxis(),2),ragdoll.getUpAxis())
        ragdoll.rightUpperLeg.tilt_str = 100
        ragdoll.rightUpperLeg.tilt_time = 20
        ragdoll.rightUpperLeg.tilt = True

        ragdoll.leftUpperLeg.final_tilt_direction =\
            add3(mul3(ragdoll.getForwardAxis(),2),ragdoll.getUpAxis())
        ragdoll.leftUpperLeg.tilt_str = 100
        ragdoll.leftUpperLeg.tilt_time = 20
        ragdoll.leftUpperLeg.tilt = True
        ragdoll.rightUpperLeg.stabilize = False
        ragdoll.leftUpperLeg.stabilize = False


    def initSitFall(ragdoll):
#    ragdoll.rightUpperLeg.stabilize = False
#    ragdoll.leftUpperLeg.stabilize = False
        ragdoll.leftUpperLeg.tilt_str = 60
        ragdoll.rightUpperLeg.tilt_str = 60
        ragdoll.leftUpperLeg.tilt_time = 1000
        ragdoll.rightUpperLeg.tilt_time = 1000
        ragdoll.leftUpperLeg.final_tilt_direction =\
            mul3(ragdoll.getForwardAxis(),1)
        ragdoll.rightUpperLeg.final_tilt_direction =\
            mul3(ragdoll.getForwardAxis(),1)


    def initSitStand(ragdoll):
        ragdoll.rightUpperLeg.tilt_str = 30
        ragdoll.rightUpperLeg.tilt_time = 1000
        ragdoll.leftUpperLeg.tilt_str = 30
        ragdoll.leftUpperLeg.tilt_time = 1000
        ragdoll.leftUpperLeg.final_tilt_direction =\
            add3(mul3(ragdoll.getForwardAxis(),0),mul3(ragdoll.getUpAxis(),-1))
        ragdoll.rightUpperLeg.final_tilt_direction =\
            add3(mul3(ragdoll.getForwardAxis(),0),mul3(ragdoll.getUpAxis(),-1))


    def sitReset(ragdoll):
        ragdoll.rightUpperLeg.stabilize = True
        ragdoll.leftUpperLeg.stabilize = True
        ragdoll.rightUpperLeg.tilt = False
        ragdoll.leftUpperLeg.tilt = False

    def initStandOnLeftLeg(ragdoll):
        """
        Stands on left leg
        """
        ragdoll.rightUpperLeg.stabilize = False
        ragdoll.rightLowerLeg.stabilize = False
        ragdoll.rightUpperLeg.tilt = False
        ragdoll.rightLowerLeg.tilt = False
        ragdoll.belly.lefttilt = True

    def initStandOnRightLeg(ragdoll):
        """
        Stands on right leg
        """
        ragdoll.leftUpperLeg.stabilize = False
        ragdoll.leftLowerLeg.stabilize = False
        ragdoll.leftUpperLeg.tilt = False
        ragdoll.leftLowerLeg.tilt = False
        ragdoll.belly.righttilt = True

    def initRestLeftLeg(ragdoll):
        ragdoll.leftUpperLeg.stabilize = True
        ragdoll.leftLowerLeg.stabilize = True
        ragdoll.leftUpperLeg.stabilizing_str = 100
        ragdoll.leftLowerLeg.stabilizing_str = 100
        ragdoll.belly.righttilt = False

    def initRestRightLeg(ragdoll):
        ragdoll.rightUpperLeg.stabilize = True
        ragdoll.rightLowerLeg.stabilize = True
        ragdoll.rightUpperLeg.stabilizing_str = 100
        ragdoll.rightLowerLeg.stabilizing_str = 100
        ragdoll.belly.lefttilt = False

    def initWalkRightLegFront(ragdoll):
        ragdoll.rightUpperLeg.tilt = True
        ragdoll.rightLowerLeg.tilt = True
        ragdoll.rightUpperLeg.tilt_time = 10
        ragdoll.rightLowerLeg.tilt_time = 10
        ragdoll.rightUpperLeg.tilt_str = 20
        ragdoll.rightLowerLeg.tilt_str = 30

        axis = ragdoll.getRelAxis(-3,0,1.75)
        ragdoll.rightUpperLeg.tilt_direction = axis
        ragdoll.rightUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-2,0,0)
        ragdoll.rightLowerLeg.tilt_direction = axis
        ragdoll.rightLowerLeg.final_tilt_direction = axis

    def initWalkLeftLegFront(ragdoll):
        ragdoll.leftUpperLeg.tilt = True
        ragdoll.leftLowerLeg.tilt = True
        ragdoll.leftUpperLeg.tilt_str = 20
        ragdoll.leftLowerLeg.tilt_str = 30
        ragdoll.leftUpperLeg.tilt_time = 10
        ragdoll.leftLowerLeg.tilt_time = 10

        axis = ragdoll.getRelAxis(-3,0,1.75)
        ragdoll.leftUpperLeg.tilt_direction = axis
        ragdoll.leftUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-2,0,0)
        ragdoll.leftLowerLeg.tilt_direction = axis
        ragdoll.leftLowerLeg.final_tilt_direction = axis

    def initPunch(ragdoll):
        ragdoll.punching = True
        ragdoll.punching = 1
        ragdoll.punch_state=1
        print "Ragdoll Started Punching"

    def initPunchRaiseArm(ragdoll):
        print "Raising Arm"
        axis = ragdoll.getRelAxis(0,3,-1)
        ragdoll.rightUpperArm.tilt_str = 30
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt_time = 10000
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 30
        ragdoll.rightForeArm.tilt_time = 10000
        axis = ragdoll.getRelAxis(3,0,1)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True

    def initPunchRaiseArm2(ragdoll):
        print "Raising Arm 2"
        axis = ragdoll.getRelAxis(0,3,-1)
        ragdoll.rightUpperArm.tilt_str = 30
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt_time = 10
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 30
        ragdoll.rightForeArm.tilt_time = 10
        axis = ragdoll.getRelAxis(1,-2,3)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True

    def initPunchExtendArm(ragdoll):
        print "Extending Arm"
        axis = ragdoll.getRelAxis(0,0,3)
        ragdoll.rightUpperArm.tilt_str = 40
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt_time = 100
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 40
        ragdoll.rightForeArm.tilt_time = 100
        axis = ragdoll.getRelAxis(0,-1,3)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.restoring_torque = 1000
        ragdoll.leftUpperLeg.stabilizing_str = 1000
        ragdoll.leftLowerLeg.stabilizing_str = 1000
        ragdoll.rightUpperLeg.stabilizing_str = 1000
        ragdoll.rightLowerLeg.stabilizing_str = 1000

    def relaxArms(ragdoll):
        ragdoll.rightUpperArm.tilt = False
        ragdoll.rightForeArm.tilt = False
        ragdoll.leftUpperArm.tilt = False
        ragdoll.leftForeArm.tilt = False
        ragdoll.leftHand.tilt = False
        ragdoll.rightHand.tilt = False


    def finishPunch(ragdoll):
        ragdoll.relaxArms()
        ragdoll.leftUpperLeg.stabilizing_str = 100
        ragdoll.leftLowerLeg.stabilizing_str = 100
        ragdoll.rightUpperLeg.stabilizing_str = 100
        ragdoll.rightLowerLeg.stabilizing_str = 100
        ragdoll.restoring_torque = 40
        ragdoll.punching = 0
        ragdoll.punch_state=1
        print "Ragdoll Stopped Punching"


    def initKick(ragdoll):
        ragdoll.kicking = 1
        ragdoll.kick_state=1
        print "Ragdoll Started Kicking"


    def initKickLegBehind(ragdoll):
        ragdoll.initStandOnLeftLeg()
        ragdoll.rightUpperLeg.tilt = True
        ragdoll.rightLowerLeg.tilt = True
        ragdoll.rightUpperLeg.tilt_str = 20
        ragdoll.rightLowerLeg.tilt_str = 30
        ragdoll.rightUpperLeg.tilt_time = 100
        ragdoll.rightLowerLeg.tilt_time = 100

        axis = ragdoll.getRelAxis(-3,0.25,-0.75)
        ragdoll.rightUpperLeg.tilt_direction = axis
        ragdoll.rightUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-2,0.5,-1)
        ragdoll.rightLowerLeg.tilt_direction = axis
        ragdoll.rightLowerLeg.final_tilt_direction = axis

    def initKickLegFront(ragdoll):
        ragdoll.initStandOnLeftLeg()
        ragdoll.rightUpperLeg.tilt = True
        ragdoll.rightLowerLeg.tilt = True
        ragdoll.rightUpperLeg.tilt_str = 20
        ragdoll.rightLowerLeg.tilt_str = 100
        ragdoll.rightUpperLeg.tilt_time = 100
        ragdoll.rightLowerLeg.tilt_time = 100

        axis = ragdoll.getRelAxis(-3,0,0.75)
        ragdoll.rightUpperLeg.tilt_direction = axis
        ragdoll.rightUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-2,0,1)
        ragdoll.rightLowerLeg.tilt_direction = axis
        ragdoll.rightLowerLeg.final_tilt_direction = axis

    def finishKick(ragdoll):
        ragdoll.initRestRightLeg()
        ragdoll.kicking = 0
        ragdoll.kick_state=1
        print "Ragdoll Finished Kicking"

    def initHandWave(ragdoll):
        ragdoll.handwaving = 1
        ragdoll.handwave_state=1
        print "Ragdoll Started Waving Hand"

    def initHandWavePos1(ragdoll):
        axis = ragdoll.getRelAxis(0,1.5,0.25)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 20
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(3,0,1)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 20
        ragdoll.rightForeArm.tilt_time = 300

    def initHandWavePos2(ragdoll):
        axis = ragdoll.getRelAxis(0,1.5,0.25)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 20
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(3,-1,1)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 20
        ragdoll.rightForeArm.tilt_time = 300


    def initHandWavePos3(ragdoll):
        axis = ragdoll.getRelAxis(0,1.5,0.25)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 2
        ragdoll.rightUpperArm.tilt_time = 3000
        axis = ragdoll.getRelAxis(3,1,0.5)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 2
        ragdoll.rightForeArm.tilt_time = 3000

    def finishHandWave(ragdoll):
        ragdoll.relaxArms()
        ragdoll.handwaving = 0
        ragdoll.handwave_state=1
        print "Ragdoll Stopped Waving Hand"

    def initNamaste(ragdoll):
        ragdoll.restoring_force = 1000
        ragdoll.namaste_on = 1
        ragdoll.namaste_state=1
        print "Ragdoll Stopped Waving Hand"

    def initNamastePos1(ragdoll):
        axis = ragdoll.getRelAxis(0,0.5,2)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 20
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(0,-0.5,2)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 20
        ragdoll.rightForeArm.tilt_time = 300

        axis = ragdoll.getRelAxis(0,-0.5,2)
        ragdoll.leftUpperArm.final_tilt_direction = axis
        ragdoll.leftUpperArm.tilt = True
        ragdoll.leftUpperArm.tilt_str = 20
        ragdoll.leftUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(0,0.5,2)
        ragdoll.leftForeArm.final_tilt_direction = axis
        ragdoll.leftForeArm.tilt = True
        ragdoll.leftForeArm.tilt_str = 20
        ragdoll.leftForeArm.tilt_time = 300

    def initNamastePos2(ragdoll):
        axis = ragdoll.getRelAxis(0,1,1)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 30
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(0,-3,1)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 30
        ragdoll.rightForeArm.tilt_time = 3000
        # Hand
        axis = ragdoll.getRelAxis(1,0,0)
        ragdoll.rightHand.final_tilt_direction = axis
        ragdoll.rightHand.tilt = True
        ragdoll.rightHand.tilt_str = 5
        ragdoll.rightHand.tilt_time = 3000

        axis = ragdoll.getRelAxis(0,-1,1)
        ragdoll.leftUpperArm.final_tilt_direction = axis
        ragdoll.leftUpperArm.tilt = True
        ragdoll.leftUpperArm.tilt_str = 30
        ragdoll.leftUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(0,3,1)
        ragdoll.leftForeArm.final_tilt_direction = axis
        ragdoll.leftForeArm.tilt = True
        ragdoll.leftForeArm.tilt_str = 30
        ragdoll.leftForeArm.tilt_time = 3000
        # Hand
        axis = ragdoll.getRelAxis(1,0,0)
        ragdoll.leftHand.final_tilt_direction = axis
        ragdoll.leftHand.tilt = True
        ragdoll.leftHand.tilt_str = 5
        ragdoll.leftHand.tilt_time = 3000

    def finishNamaste(ragdoll):
        ragdoll.relaxArms()
        ragdoll.restoring_force = 40
        ragdoll.namaste_on = 0
        ragdoll.namaste_state=1
        print "Ragdoll Stopped Waving Hand"
        
    def initJump(ragdoll):
        ragdoll.jumping = 1
        ragdoll.jump_state=1
        print "Ragdoll Started Jumping"
        
    def initPrepareForJump(ragdoll):
        #ragdoll.rightUpperLeg.tilt = True
        ragdoll.rightLowerLeg.tilt = True
        ragdoll.rightUpperLeg.tilt_str = 100
        ragdoll.rightLowerLeg.tilt_str = 100
        ragdoll.rightUpperLeg.tilt_time = 100
        ragdoll.rightLowerLeg.tilt_time = 100

        #axis = ragdoll.getRelAxis(3,0.25,0.75)
        #ragdoll.rightUpperLeg.tilt_direction = axis
        #ragdoll.rightUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-3,0,-1)
        ragdoll.rightLowerLeg.tilt_direction = axis
        ragdoll.rightLowerLeg.final_tilt_direction = axis
        
        #ragdoll.leftUpperLeg.tilt = True
        ragdoll.leftLowerLeg.tilt = True
        ragdoll.leftUpperLeg.tilt_str = 100
        ragdoll.leftLowerLeg.tilt_str = 100
        ragdoll.leftUpperLeg.tilt_time = 100
        ragdoll.leftLowerLeg.tilt_time = 100

        #axis = ragdoll.getRelAxis(3,-0.25,0.75)
        #ragdoll.leftUpperLeg.tilt_direction = axis
        #ragdoll.leftUpperLeg.final_tilt_direction = axis

        axis = ragdoll.getRelAxis(-3,0,-1)
        ragdoll.leftLowerLeg.tilt_direction = axis
        ragdoll.leftLowerLeg.final_tilt_direction = axis
        
    def initJumpInAir(ragdoll):
        ragdoll.pelvis.addForce(mul3((0,1,0),6000))

    def finishJump(ragdoll):
        ragdoll.leftLowerLeg.tilt = False
        ragdoll.rightLowerLeg.tilt = False
        ragdoll.jumping = 0
        ragdoll.jump_state=1
        print "Ragdoll Stopped Jumping"


    def initHandshake(ragdoll):
        ragdoll.handshaking = 1
        ragdoll.handshake_state=1
        print "Ragdoll Started Shaking Hands"
        
    def initHandshakePos1(ragdoll):
        axis = ragdoll.getRelAxis(-3,0,2)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 10
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(-0.2,-2,3)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 50
        ragdoll.rightForeArm.tilt_time = 3000
        
        axis = ragdoll.getRelAxis(1,-1,0)
        ragdoll.rightHand.final_tilt_direction = axis
        ragdoll.rightHand.tilt = True
        ragdoll.rightHand.tilt_str = 5
        ragdoll.rightHand.tilt_time = 3000
    
    def initHandshakePos2(ragdoll):
        axis = ragdoll.getRelAxis(-3,0,2)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 10
        ragdoll.rightUpperArm.tilt_time = 300
        axis = ragdoll.getRelAxis(2,-2,3)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.rightForeArm.tilt_str = 30
        ragdoll.rightForeArm.tilt_time = 3000
        
        axis = ragdoll.getRelAxis(1,-1,0)
        ragdoll.rightHand.final_tilt_direction = axis
        ragdoll.rightHand.tilt = True
        ragdoll.rightHand.tilt_str = 5
        ragdoll.rightHand.tilt_time = 3000
        
    def finishHandshake(ragdoll):
        ragdoll.relaxArms()
        ragdoll.handshaking = 0
        ragdoll.handshake_state=1
        print "Ragdoll Stopped Shaking Hands"
                
def createCube(world,space,density,length,color=(0.8,0.8,0.8)):
    """Creates a cube body and corresponding geom."""
    body = ode.Body(world)
    body.color = color
    M = ode.Mass()
    M.setBox(density,length,length,length)
    body.setMass(M)
    body.shape = 'cube'
    body.length = length
    geom = ode.GeomBox(space,(length,length,length))
    geom.setBody(body)
    return body,geom

def createBall(world,space,density,radius,color=(0.8,0.8,0.8)):
    """Creates a spherical body and corresponding geom."""
    body = ode.Body(world)
    body.color = color
    M = ode.Mass()
    M.setSphere(density,radius)
    body.setMass(M)
    body.shape = 'sphere'
    body.radius = radius
    geom = ode.GeomSphere(space,radius)
    geom.setBody(body)
    return body,geom

def createCapsule(world, space, density, length, radius,color=(0.8,0.8,0.8)):
    """Creates a capsule body and corresponding geom."""

    # create capsule body (aligned along the z-axis so that it matches the
    #   GeomCCylinder created below, which is aligned along the z-axis by
    #   default)
    body = ode.Body(world)
    body.color = color
    M = ode.Mass()
    M.setCylinder(density, 3, radius, length)
    body.setMass(M)

    # set parameters for drawing the body
    body.shape = "capsule"
    body.length = length
    body.radius = radius

    # create a capsule geom for collision detection
    geom = ode.GeomCCylinder(space, radius, length)
    geom.setBody(body)

    return body, geom

def near_callback(args, geom1, geom2):
    """
    Callback function for the collide() method.

    This function checks if the given geoms do collide and creates contact
    joints if they do.
    """

    if (ode.areConnected(geom1.getBody(), geom2.getBody())):
        return

    # check if the objects collide
    contacts = ode.collide(geom1, geom2)

    # create contact joints
    world, contactgroup = args
    for c in contacts:
        c.setBounce(0.2)
        c.setMu(500) # 0-5 = very slippery, 50-500 = normal, 5000 = very sticky
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())
    b1= geom1.getBody()
    b2= geom2.getBody()
    #if b1:
        #print b1.tag
    #if b2:
        #print b2.tag
    #print '----'

def prepare_GL():
    """Setup basic OpenGL rendering with smooth shading and a single light."""

    glClearColor(0.8, 0.8, 0.9, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective (45.0, 1.3333, 0.2, 20.0)

    glViewport(0, 0, 640, 480)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glLightfv(GL_LIGHT0,GL_POSITION,[0, 0, 1, 0])
    glLightfv(GL_LIGHT0,GL_DIFFUSE,[1, 1, 1, 1])
    glLightfv(GL_LIGHT0,GL_SPECULAR,[1, 1, 1, 1])
    glEnable(GL_LIGHT0)

    glEnable(GL_COLOR_MATERIAL)
    glColor3f(0.8, 0.8, 0.8)

    gluLookAt(eye[0],eye[1],eye[2],look_obj[0],look_obj[1],look_obj[2],
            look_up[0],look_up[1],look_up[2])

# polygon resolution for capsule bodies

CAPSULE_SLICES = 16
CAPSULE_STACKS = 12

def draw_floor():

    glBegin(GL_LINES)
    for j in xrange(100):
        i = j/5.0
        if i==0:
            glColor3f(.6,.3,.3)
        else:
            glColor3f(.25,.25,.25)
        glVertex3f(i-10,0,-10)
        glVertex3f(i-10,0,20-10)
        if i==0:
            glColor3f(.3,.3,.6)
        else:
            glColor3f(.25,.25,.25)
        glVertex3f(-10,0,i-10)
        glVertex3f(20-10,0,i-10)
    glEnd()
    glColor3f(0.8, 0.8, 0.8)

def draw_body(body):
    """Draw an ODE body."""

    rot = makeOpenGLMatrix(body.getRotation(), body.getPosition())
    glPushMatrix()
    glMultMatrixd(rot)
    glColor3f(body.color[0],body.color[1],body.color[2])
    if body.shape == "capsule":
        cylHalfHeight = body.length / 2.0
        glBegin(GL_QUAD_STRIP)
        for i in range(0, CAPSULE_SLICES + 1):
            angle = i / float(CAPSULE_SLICES) * 2.0 * pi
            ca = cos(angle)
            sa = sin(angle)
            glNormal3f(ca, sa, 0)
            glVertex3f(body.radius * ca, body.radius * sa, cylHalfHeight)
            glVertex3f(body.radius * ca, body.radius * sa, -cylHalfHeight)
        glEnd()
        glTranslated(0, 0, cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
        glTranslated(0, 0, -2.0 * cylHalfHeight)
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)
    elif body.shape == "cube":
        glutSolidCube(body.length)
    elif body.shape == "sphere":
        glutSolidSphere(body.radius, CAPSULE_SLICES, CAPSULE_STACKS)

    glPopMatrix()




def onKey(c, x, y):
    """GLUT keyboard callback."""

    global SloMo, Paused,ragdoll,curr_ragdoll

    # set simulation speed
#    if c >= '0' and c <= '9':
#        SloMo = 4 * int(c) + 1
    # pause/unpause simulation
    if c == 'p' or c == 'P':
        Paused = not Paused
    # quit
    elif c == 'q' or c == 'Q':
        sys.exit(0)

    elif c == 'b':
        obstacle, obsgeom = createCapsule(world, space, 1000, 0.05, 0.15)
        diff = (random.uniform(-1, 1), 0, random.uniform(-1, 1))
        target = ragdoll.chest.getPosition()
        pos = add3(target,diff)
        obstacle.setPosition(pos)
        obstacle.setLinearVel(mul3(diff,-50))
        obstacle.setRotation(rightRot)
        bodies.append(obstacle)
        geoms.append(obsgeom)
        print "obstacle created at %s" % (str(pos))

    elif c == 'm':
        ragdoll.rightHand.destination = ragdoll.getRelPos(0.5,0.5,0.5)
        ragdoll.rightHand.moving = True
        
    elif c == 'j':
        ragdoll.initJump()

    elif c == 't':
        ragdoll.initHandshake()
    elif c == 'T':
        ragdoll.finishHandshake()

    elif c == 'h':
        ragdoll.initPunch()

    elif c == 'w':
        ragdoll.initHandWave()
    elif c == 'e':
        ragdoll.finishHandWave()

    elif c == 'W':
        ragdoll.walking = 1
        ragdoll.walk_state=1
        print "Ragdoll Started Walking"
    elif c == 'Z':
        ragdoll.walking = 0
        ragdoll.walk_state=1
        print "Ragdoll Stopped Walking"
        ragdoll.initRestRightLeg()
        ragdoll.initRestLeftLeg()

    elif c == 'a':
        print "Bending Arm"
        axis = ragdoll.getRelAxis(-3,1.5,1)
        ragdoll.rightUpperArm.final_tilt_direction = axis
        ragdoll.rightUpperArm.tilt = True
        ragdoll.rightUpperArm.tilt_str = 30
        ragdoll.rightUpperArm.tilt_time = 100

        axis = ragdoll.getRelAxis(-3,0,1)
        ragdoll.rightForeArm.final_tilt_direction = axis
        ragdoll.rightForeArm.tilt = True
        ragdoll.leftForeArm.tilt_str = 30
        ragdoll.rightForeArm.tilt_time = 100

        axis = ragdoll.getRelAxis(-3,-1.5,1)
        ragdoll.leftUpperArm.final_tilt_direction = axis
        ragdoll.leftUpperArm.tilt = True
        ragdoll.leftUpperArm.tilt_str = 30
        ragdoll.leftUpperArm.tilt_time = 100
        axis = ragdoll.getRelAxis(-3,0,1)
        ragdoll.leftForeArm.final_tilt_direction = axis
        ragdoll.leftForeArm.tilt = True
        ragdoll.leftForeArm.tilt_str = 30
        ragdoll.leftForeArm.tilt_time = 100

    elif c=='c':
        print "Releasing Arm"
        ragdoll.relaxArms()

    elif c=='l':
        ragdoll.kickass()

    elif c == 'B':
        print 'Made block'
        body, geom = createCube(world,space,100000,0.45)
        bodies.append(body)
        geoms.append(geom)
        rp = ragdoll.pelvis.getPosition()
        blockpos = reduce(add3, [mul3(ragdoll.getUpAxis(),-0.6),
                mul3(ragdoll.getForwardAxis(),-0.36),rp])
        body.setPosition(blockpos)
        body.setRotation(ragdoll.pelvis.getRotation())
    elif c == 'k':
        body, geom = createBall(world,space,0.01,0.15)
        lp = ragdoll.rightLowerLeg.getPosition()
        body.setPosition(reduce(add3,[lp,mul3(ragdoll.getForwardAxis(),0.31),
            mul3(ragdoll.getUpAxis(),-0.0),mul3(ragdoll.getRightAxis(),-0.11)]))
        bodies.append(body)
        geoms.append(geom)
        ragdoll.initKick()

    elif c == 's':
        ragdoll.sitting = 1
        ragdoll.sit_state = 1
    elif c == 'n':
        ragdoll.initNamaste()

    elif c == '+':
        direction = norm3(sub3(ragdoll.belly.getPosition(),eye))
        tmp_eye = add3(eye,mul3(direction,0.1))
        tmp_look_obj = add3(look_obj,mul3(direction,0.1))
        for i in xrange(3):
            eye[i]=tmp_eye[i]
            look_obj[i]=tmp_look_obj[i]

    elif c == '-':
        direction = norm3(sub3(ragdoll.belly.getPosition(),eye))
        tmp_eye = add3(eye,mul3(direction,-0.1))
        tmp_look_obj = add3(look_obj,mul3(direction,-0.1))
        for i in xrange(3):
            eye[i]=tmp_eye[i]
            look_obj[i]=tmp_look_obj[i]
    elif c == '2':
        curr_ragdoll = (curr_ragdoll+1)%len(ragdolls)
        ragdoll = ragdolls[curr_ragdoll]
    elif c == 'o':
        if ragdoll:
            ragdoll = None
        ragdolls.pop()
    elif c == 'O':
        ragdolls.append(RagDoll(world, space, 500, (0.0, 0.9, 0.0)))
        ragdoll = ragdolls[0]

def onDraw():
    """GLUT render callback."""

    prepare_GL()

    draw_floor()
    for b in bodies:
        draw_body(b)
    for ragdoll in ragdolls:
        for b in ragdoll.bodies:
            draw_body(b)
#    if ragdoll.rightHand.moving:
#        dst = ragdoll.rightHand.destination
#        glTranslated(dst[0],dst[1],dst[2]);
#        color = [0.5,0.8,0.1]
#        glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
#        glutSolidSphere(0.1,10,10)

    glutSwapBuffers()


def onIdle():
    """GLUT idle processing callback, performs ODE simulation step."""

    global Paused, lasttime, numiter

    if Paused:
        return

    t = dt - (time.time() - lasttime)
    if (t > 0):
        time.sleep(t)

    glutPostRedisplay()

    for i in range(stepsPerFrame):
        # Detect collisions and create contact joints
        space.collide((world, contactgroup), near_callback)

        # Simulation step (with slo motion)
        world.step(dt / stepsPerFrame / SloMo)

        numiter += 1

        # apply internal ragdoll forces
        for ragdoll in ragdolls:
            ragdoll.update()

        # Remove all contact joints
        contactgroup.empty()

    lasttime = time.time()

def processSpecialKeys(key, xx, yy):
    if key == GLUT_KEY_LEFT :
        eye[0]-=0.1
        look_obj[0]-=0.1
    elif key == GLUT_KEY_RIGHT :
        eye[0]+=0.1
        look_obj[0]+=0.1
    elif key == GLUT_KEY_DOWN :
        eye[1]-=0.1
        look_obj[1]-=0.1
    elif key == GLUT_KEY_UP :
        eye[1]+=0.1
        look_obj[1]+=0.1


# initialize GLUT
glutInit([])
glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)

# create the program window
x = 0
y = 0
width = 640
height = 480
glutInitWindowPosition(x, y);
glutInitWindowSize(width, height);
glutCreateWindow("PyODE Ragdoll Simulation")

# create an ODE world object
world = ode.World()
world.setGravity((0.0, -9.81, 0.0))
#world.setGravity((0.0, -0.00, 0.0))
world.setERP(0.1)
world.setCFM(1E-4)

# create an ODE space object
space = ode.Space()

# create a plane geom to simulate a floor
floor = ode.GeomPlane(space, (0, 1, 0), 0)

# create a list to store any ODE bodies which are not part of the ragdoll (this
#   is needed to avoid Python garbage collecting these bodies)
bodies = []

geoms = []

# create a joint group for the contact joints generated during collisions
#   between two bodies collide
contactgroup = ode.JointGroup()

# set the initial simulation loop parameters
fps = 60
dt = 1.0 / fps
stepsPerFrame = 20
SloMo = 1
Paused = False
lasttime = time.time()
numiter = 0

# create the ragdoll
ragdolls=[]
ragdolls.append(RagDoll(world, space, 500, (0.0, 0.9, 0.0)))

#ragdolls.append(RagDoll(world, space, 500, (0.4, 0.9, 0.4)))
#print "total mass is %.1f kg (%.1f lbs)" % (ragdoll2.totalMass,
#    ragdoll.totalMass * 2.2)

curr_ragdoll = 0
ragdoll = ragdolls[0]

# create an obstacle
#obstacle, obsgeom = createCapsule(world, space, 1000, 0.05, 0.15)
#pos = (random.uniform(-0.3, 0.3), 0.2, random.uniform(-0.15, 0.2))
##pos = (0.27396178783269359, 0.20000000000000001, 0.17531818795388002)
#obstacle.setPosition(pos)
#obstacle.setRotation(rightRot)
#bodies.append(obstacle)
#print "obstacle created at %s" % (str(pos))

# set GLUT callbacks
glutKeyboardFunc(onKey)
glutDisplayFunc(onDraw)
glutIdleFunc(onIdle)
glutSpecialFunc(processSpecialKeys)
#walking state global
walking = 0
#Camera position globals
eye = [1.5,6.0,6.0]
look_obj = [0.0,1.0,0.0]
look_up = [0,1,0]

# enter the GLUT event loop
glutMainLoop()
