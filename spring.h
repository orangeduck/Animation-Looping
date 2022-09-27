#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"

//--------------------------------------

static inline float damper_exact(float x, float g, float halflife, float dt, float eps=1e-5f)
{
    return lerpf(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline vec3 damper_exact(vec3 x, vec3 g, float halflife, float dt, float eps=1e-5f)
{
    return lerp(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline quat damper_exact(quat x, quat g, float halflife, float dt, float eps=1e-5f)
{
    return quat_slerp_shortest_approx(x, g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline float damp_adjustment_exact(float g, float halflife, float dt, float eps=1e-5f)
{
    return g * (1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline vec3 damp_adjustment_exact(vec3 g, float halflife, float dt, float eps=1e-5f)
{
    return g * (1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

static inline quat damp_adjustment_exact(quat g, float halflife, float dt, float eps=1e-5f)
{
    return quat_slerp_shortest_approx(quat(), g, 1.0f - fast_negexpf((LN2f * dt) / (halflife + eps)));
}

//--------------------------------------

static inline float halflife_to_damping(float halflife, float eps = 1e-5f)
{
    return (4.0f * LN2f) / (halflife + eps);
}
    
static inline float damping_to_halflife(float damping, float eps = 1e-5f)
{
    return (4.0f * LN2f) / (damping + eps);
}

static inline float frequency_to_stiffness(float frequency)
{
   return squaref(2.0f * PIf * frequency);
}

static inline float stiffness_to_frequency(float stiffness)
{
    return sqrtf(stiffness) / (2.0f * PIf);
}

//--------------------------------------

static inline void simple_spring_damper_exact(
    float& x, 
    float& v, 
    const float x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    float j0 = x - x_goal;
    float j1 = v + j0*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(j0 + j1*dt) + x_goal;
    v = eydt*(v - j1*y*dt);
}

static inline void simple_spring_damper_exact(
    vec3& x, 
    vec3& v, 
    const vec3 x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    vec3 j0 = x - x_goal;
    vec3 j1 = v + j0*y;
    float eydt = fast_negexpf(y*dt);

    x = eydt*(j0 + j1*dt) + x_goal;
    v = eydt*(v - j1*y*dt);
}

static inline void simple_spring_damper_exact(
    quat& x, 
    vec3& v, 
    const quat x_goal, 
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
	
    vec3 j0 = quat_to_scaled_angle_axis(quat_abs(quat_mul(x, quat_inv(x_goal))));
    vec3 j1 = v + j0*y;
	
    float eydt = fast_negexpf(y*dt);

    x = quat_mul(quat_from_scaled_angle_axis(eydt*(j0 + j1*dt)), x_goal);
    v = eydt*(v - j1*y*dt);
}

//--------------------------------------

