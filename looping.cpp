extern "C"
{
#include "raylib.h"
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
}
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "spring.h"
#include "array.h"
#include "character.h"
#include "database.h"

#include <initializer_list>
#include <vector>
#include <functional>
#include <string>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const float mouse_dx,
    const float dt)
{
    return azimuth + 1.0f * dt * -mouse_dx;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const float mouse_dy,
    const float dt)
{
    return clampf(altitude + 1.0f * dt * mouse_dy, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    return clampf(distance +  20.0f * dt * -GetMouseWheelMove(), 0.1f, 100.0f);
}

void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const float mouse_dx,
    const float mouse_dy,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, mouse_dx, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, mouse_dy, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
    
    UpdateCamera(&cam);
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

//--------------------------------------

void compute_start_end_positional_difference(
    slice1d<vec3> diff_pos,
    slice1d<vec3> diff_vel,
    const slice2d<vec3> pos,
    const float dt)
{
    // Check we have at least 2 frames of animation
    assert(pos.rows >= 2);

    // Loop over every joint
    for (int j = 0; j < pos.cols; j++)
    {
        // Positional difference between first and last frame
        diff_pos(j) = pos(pos.rows-1, j) - pos(0, j);
        
        // Velocity difference between first and last frame
        diff_vel(j) = 
            ((pos(pos.rows-1, j) - pos(pos.rows-2, j)) / dt) -
            ((pos(         1, j) - pos(         0, j)) / dt);
    }
}

void compute_start_end_rotational_difference(
    slice1d<vec3> diff_rot,
    slice1d<vec3> diff_vel,
    const slice2d<quat> rot,
    const float dt)
{
    // Check we have at least 2 frames of animation
    assert(rot.rows >= 2);

    // Loop over every joint
    for (int j = 0; j < rot.cols; j++)
    {
        // Rotational difference between first and last frame 
        // represented in scaled-angle-axis space
        diff_rot(j) = quat_to_scaled_angle_axis(
            quat_abs(quat_mul_inv(rot(rot.rows-1, j), rot(0, j))));

        // Angular velocity difference between first and last frame
        diff_vel(j) = 
            quat_differentiate_angular_velocity(
                rot(rot.rows-1, j), rot(rot.rows-2, j), dt) -
            quat_differentiate_angular_velocity(
                rot(         1, j), rot(         0, j), dt);
    }
}

//--------------------------------------

void apply_positional_offsets(
    slice2d<vec3> out, 
    const slice2d<vec3> pos, 
    const slice2d<vec3> offsets)
{
    // Loop over every frame
    for (int i = 0; i < pos.rows; i++)
    {
        // Loop over every joint
        for (int j = 0; j < pos.cols; j++)
        {
            // Simply add on offset
            out(i, j) = pos(i, j) + offsets(i, j);
        }
    }
}

void apply_rotational_offsets(
    slice2d<quat> out, 
    const slice2d<quat> rot, 
    const slice2d<vec3> offsets)
{
    // Loop over every frame
    for (int i = 0; i < rot.rows; i++)
    {
        // Loop over every joint
        for (int j = 0; j < rot.cols; j++)
        {
            // Convert back from scaled-angle-axis space and
            // multiply on the left. This rotates the first
            // frame toward the last frame.
            out(i, j) = quat_mul(
                quat_from_scaled_angle_axis(offsets(i, j)),
                rot(i, j));
        }
    }
}

//--------------------------------------

vec3 decayed_offset(
    const vec3 x, // Initial Position
    const vec3 v, // Initial Velocity
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    vec3 j1 = v + x*y;
    float eydt = fast_negexpf(y*dt);

    return eydt*(x + j1*dt);
}

void compute_inertialize_start_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float halflife, 
    const float dt)
{
    for (int i = 0; i < offsets.rows; i++)
    {
        for (int j = 0; j < offsets.cols; j++)
        {
            offsets(i, j) = decayed_offset(
                diff_pos(j), 
                diff_vel(j), 
                halflife, 
                i * dt);
        }
    }
}

void compute_inertialize_end_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float halflife, 
    const float dt)
{
    for (int i = 0; i < offsets.rows; i++)
    {
        for (int j = 0; j < offsets.cols; j++)
        {
            offsets(i, j) = decayed_offset(
                -diff_pos(j), 
                diff_vel(j), 
                halflife, 
                ((offsets.rows-1) - i) * dt);
        }
    }
}

void compute_inertialize_both_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float halflife_start, 
    const float halflife_end, 
    const float ratio,
    const float dt)
{
    // Check ratio of correction for start
    // and end is between 0 and 1
    assert(ratio >= 0.0f && ratio <= 1.0f);
  
    // Loop over every frame
    for (int i = 0; i < offsets.rows; i++)
    {
        // Loop over every joint
        for (int j = 0; j < offsets.cols; j++)
        {
            offsets(i, j) = 
                // Decayed offset from start
                decayed_offset(
                    ratio * diff_pos(j), 
                    ratio * diff_vel(j), 
                    halflife_start, 
                    i * dt) +
                // Decayed offset from end
                decayed_offset(
                    (1.0f-ratio) * -diff_pos(j), 
                    (1.0f-ratio) *  diff_vel(j), 
                    halflife_end, 
                    ((offsets.rows-1) - i) * dt);
        }
    }
}

//--------------------------------------

vec3 decayed_offset_cubic(
    const vec3 x, // Initial Position
    const vec3 v, // Initial Velocity
    const float blendtime, 
    const float dt)
{
    float t = clampf(dt / blendtime, 0, 1);

    vec3 d = x;
    vec3 c = v * blendtime;
    vec3 b = -3*d - 2*c;
    vec3 a = 2*d + c;
    
    return a*t*t*t + b*t*t + c*t + d;
}

void compute_inertialize_cubic_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float blendtime_start, 
    const float blendtime_end, 
    const float ratio,
    const float dt)
{
    assert(ratio >= 0.0f && ratio <= 1.0f);
    
    for (int i = 0; i < offsets.rows; i++)
    {
        for (int j = 0; j < offsets.cols; j++)
        {
            offsets(i, j) = 
                decayed_offset_cubic(
                    ratio * diff_pos(j), 
                    ratio * diff_vel(j), 
                    blendtime_start, 
                    i * dt) +
                decayed_offset_cubic(
                    (1.0f-ratio) * -diff_pos(j), 
                    (1.0f-ratio) *  diff_vel(j), 
                    blendtime_end, 
                    ((offsets.rows-1) - i) * dt);
        }
    }
}

//--------------------------------------

vec3 decayed_velocity_offset(
    const vec3 v, // Initial Velocity
    const float halflife, 
    const float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    return fast_negexpf(y*dt)*v*dt;
}

vec3 decayed_velocity_offset_cubic(
    const vec3 v, // Initial Velocity 
    const float blendtime, 
    const float dt)
{
    float t = clampf(dt / blendtime, 0, 1);

    vec3 c = v * blendtime;
    vec3 b = -2*c;
    vec3 a = c;
    
    return a*t*t*t + b*t*t + c*t;
}

void compute_linear_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const float ratio)
{
    for (int i = 0; i < offsets.rows; i++)
    {
        for (int j = 0; j < offsets.cols; j++)
        {    
            offsets(i, j) = lerpf(
                 ratio,
                (ratio-1.0f),
                ((float)i / (offsets.rows-1))) * diff_pos(j);
        }
    }
}

void compute_linear_inertialize_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float blendtime_start, 
    const float blendtime_end, 
    const float ratio,
    const float dt)
{
    // Input sanity checks
    assert(ratio >= 0.0f && ratio <= 1.0f);
    assert(blendtime_start >= 0.0f);
    assert(blendtime_end >= 0.0f);

    // Loop over every frame
    for (int i = 0; i < offsets.rows; i++)
    {
        // Loop over every joint
        for (int j = 0; j < offsets.cols; j++)
        {    
            offsets(i, j) = 
                // Initial linear offset
                lerpf(
                     ratio,
                    (ratio-1.0f),
                    ((float)i / (offsets.rows-1))) * diff_pos(j) + 
                // Velocity offset at start
                decayed_velocity_offset_cubic(
                    ratio * diff_vel(j), 
                    blendtime_start, 
                    i * dt) + 
                // Velocity offset at end
                decayed_velocity_offset_cubic(
                    (1.0f-ratio) * diff_vel(j), 
                    blendtime_end, 
                    ((offsets.rows-1) - i) * dt);
        }
    }
}

//--------------------------------------

float softfade(const float x, const float alpha)
{
    return logf(1.0f + expf(alpha - 2.0f*alpha*x)) / alpha;
}

// Function using `softfade` to decay some offset
vec3 decayed_offset_softfade(
    const vec3 x, // Initial Position
    const float duration,
    const float hardness,
    const float dt)
{
    return x * softfade(dt / duration, hardness);
}

// Gradient of the `softfade` function at zero
float softfade_grad_zero(const float alpha)
{
    return (-2.0f * expf(alpha)) / (1.0f + expf(alpha));
}

// Gradient of the `decayed_offset_softfade` 
// function with a `dt` of zero
vec3 decayed_offset_softfade_grad_zero(
    const vec3 x,
    const float duration,
    const float hardness)
{
    return x * (softfade_grad_zero(hardness) / duration);
}

void compute_softfade_start_end_difference(
    slice1d<vec3> diff_pos,
    slice1d<vec3> diff_vel,
    const slice2d<vec3> pos,
    const float duration_start, 
    const float duration_end, 
    const float hardness_start, 
    const float hardness_end, 
    const float ratio,
    const float dt)
{
    assert(pos.rows >= 2);

    // Loop over every joint
    for (int j = 0; j < pos.cols; j++)
    {
        // Positional difference between first and last frame
        diff_pos(j) = pos(pos.rows-1, j) - pos(0, j);
        
        // End frame velocity (including softfade)
        vec3 velocity_end = 
            (pos(pos.rows-1, j) - pos(pos.rows-2, j)) / dt + 
            decayed_offset_softfade_grad_zero(
                ratio * diff_pos(j), 
                duration_start, 
                hardness_start) * dt;
        
        // Start frame velocity (including softfade)
        vec3 velocity_start = 
            (pos(         1, j) - pos(         0, j)) / dt +
            decayed_offset_softfade_grad_zero(
                (1.0f-ratio) * diff_pos(j), 
                duration_end, 
                hardness_end) * dt;

        // Velocity difference between first and last frame
        diff_vel(j) = velocity_end - velocity_start;
    }
}

void compute_softfade_start_end_difference(
    slice1d<vec3> diff_rot,
    slice1d<vec3> diff_vel,
    const slice2d<quat> rot,
    const float duration_start, 
    const float duration_end, 
    const float hardness_start, 
    const float hardness_end, 
    const float ratio,
    const float dt)
{
    assert(rot.rows >= 2);

    // Loop over every joint
    for (int j = 0; j < rot.cols; j++)
    {
        // Rotational difference between first and last frame 
        // represented in scaled-angle-axis space
        diff_rot(j) = quat_to_scaled_angle_axis(
            quat_abs(quat_mul_inv(rot(rot.rows-1, j), rot(0, j))));
        
        // End frame velocity (including softfade)
        vec3 velocity_end = 
            quat_differentiate_angular_velocity(
                rot(rot.rows-1, j), rot(rot.rows-2, j), dt) + 
            decayed_offset_softfade_grad_zero(
                ratio * diff_rot(j), 
                duration_start, 
                hardness_start) * dt;
        
        // Start frame velocity (including softfade)
        vec3 velocity_start = 
            quat_differentiate_angular_velocity(
                rot(         1, j), rot(         0, j), dt) +
            decayed_offset_softfade_grad_zero(
                (1.0f-ratio) * diff_rot(j), 
                duration_end, 
                hardness_end) * dt;

        // Velocity difference between first and last frame
        diff_vel(j) = velocity_end - velocity_start;
    }
}

void compute_softfade_inertialize_offsets(
    slice2d<vec3> offsets,
    const slice1d<vec3> diff_pos,
    const slice1d<vec3> diff_vel,
    const float blendtime_start, 
    const float blendtime_end, 
    const float duration_start, 
    const float duration_end, 
    const float hardness_start, 
    const float hardness_end, 
    const float ratio,
    const float dt)
{
    // Loop over every frame
    for (int i = 0; i < offsets.rows; i++)
    {
        // Loop over every joint
        for (int j = 0; j < offsets.cols; j++)
        {    
            offsets(i, j) = 
                // Softfade at start
                decayed_offset_softfade(
                    ratio * diff_pos(j), 
                    duration_start, 
                    hardness_start, 
                    i * dt) + 
                // Softfade at end
                decayed_offset_softfade(
                    (1.0f-ratio) * -diff_pos(j), 
                    duration_end, 
                    hardness_end, 
                    ((offsets.rows-1) - i) * dt) + 
                // Velocity offset at start
                decayed_velocity_offset_cubic(
                    ratio * diff_vel(j), 
                    blendtime_start, 
                    i * dt) + 
                // Velocity offset at end
                decayed_velocity_offset_cubic(
                    (1.0f-ratio) * diff_vel(j), 
                    blendtime_end, 
                    ((offsets.rows-1) - i) * dt);
        }
    }
}

//--------------------------------------

void compute_root_inertialize_offsets(
    slice2d<vec3> offsets_pos, 
    slice2d<vec3> offsets_rot, 
    const slice2d<vec3> pos, 
    const slice2d<quat> rot, 
    const float blendtime_start, 
    const float blendtime_end, 
    const float ratio,
    const float dt)
{
    // Check animation is at least 2 frames
    assert(rot.rows >= 2 && pos.rows >= 2);
    
    // Get root start and end rotations
    quat root_start = rot(         0, 0);
    quat root_end   = rot(rot.rows-1, 0);
    
    // Compute character space difference in positional velocity
    vec3 pos_vel_end   = quat_inv_mul_vec3(root_end, 
        (pos(pos.rows-1, 0) - pos(pos.rows-2, 0)) / dt);
    vec3 pos_vel_start = quat_inv_mul_vec3(root_start, 
        (pos(         1, 0) - pos(         0, 0)) / dt);
    vec3 diff_pos_vel = pos_vel_end - pos_vel_start;
    
    // Compute character space difference in rotational velocity
    vec3 rot_vel_end   = quat_inv_mul_vec3(root_end, 
        quat_differentiate_angular_velocity(
            rot(rot.rows-1, 0), rot(rot.rows-2, 0), dt));
    vec3 rot_vel_start = quat_inv_mul_vec3(root_start,  
         quat_differentiate_angular_velocity(
            rot(         1, 0), rot(         0, 0), dt));
    vec3 diff_rot_vel = rot_vel_end - rot_vel_start;
    
    // Loop over frames
    for (int i = 0; i < rot.rows; i++)
    {
        // Root positional offset
        offsets_pos(i, 0) = 
            // Velocity offset at start
            decayed_velocity_offset_cubic(
                ratio * quat_mul_vec3(root_start, diff_pos_vel),
                blendtime_start,
                i * dt) +
            // velocity offset at end
            decayed_velocity_offset_cubic(
                (1.0f-ratio) * quat_mul_vec3(root_end, diff_pos_vel),
                blendtime_end,
                ((rot.rows-1) - i) * dt);
        
        // Root rotational offset
        offsets_rot(i, 0) =
            // Velocity offset at start
            decayed_velocity_offset_cubic(
                ratio * quat_mul_vec3(root_start, diff_rot_vel),
                blendtime_start,
                i * dt) +
            // velocity offset at end
            decayed_velocity_offset_cubic(
                (1.0f-ratio) * quat_mul_vec3(root_end, diff_rot_vel),
                blendtime_end,
                ((rot.rows-1) - i) * dt);
    }
}

//--------------------------------------

static inline void animation_sample(
    slice1d<vec3> sampled,
    const slice2d<vec3> pos,
    const float time,
    const float dt)
{
    int st = (int)(time / dt);
    int s0 = clamp(st + 0, 0, pos.rows - 1); 
    int s1 = clamp(st + 1, 0, pos.rows - 1);
    float alpha = fmod(time / dt, 1.0f);
    
    for (int j = 0; j < pos.cols; j++)
    {
        sampled(j) = lerp(pos(s0, j), pos(s1, j), alpha);
    }
}

static inline void animation_sample(
    slice1d<quat> sampled,
    const slice2d<quat> rot,
    const float time,
    const float dt)
{
    int st = (int)(time / dt);
    int s0 = clamp(st + 0, 0, rot.rows - 1); 
    int s1 = clamp(st + 1, 0, rot.rows - 1);
    float alpha = fmod(time / dt, 1.0f);
    
    for (int j = 0; j < rot.cols; j++)
    {
        sampled(j) = quat_nlerp(rot(s0, j), rot(s1, j), alpha);
    }
}

static inline void animation_sample_root_local_velocity(
    vec3& sampled_root_velocity, 
    vec3& sampled_root_angular_velocity,
    const slice2d<vec3> pos,
    const slice2d<quat> rot,
    const float time,
    const float dt)
{
    int s0 = clamp((int)(time / dt), 0, pos.rows - 1); 
    float alpha = fmod(time / dt, 1.0f);
    
    if (s0 == 0)
    {
        sampled_root_velocity = quat_inv_mul_vec3(
            rot(s0, 0),
            (pos(s0 + 1, 0) - pos(s0 + 0, 0)) / dt);
            
        sampled_root_angular_velocity = quat_inv_mul_vec3(
            rot(s0, 0),
            quat_differentiate_angular_velocity(rot(s0 + 1, 0), rot(s0 + 0, 0), dt));
    }
    else
    {
        sampled_root_velocity = quat_inv_mul_vec3(
            rot(s0, 0),
            (pos(s0 - 0, 0) - pos(s0 - 1, 0)) / dt);
        
        sampled_root_angular_velocity = quat_inv_mul_vec3(
            rot(s0, 0),
            quat_differentiate_angular_velocity(rot(s0 - 0, 0), rot(s0 - 1, 0), dt));
    }
}                

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

enum
{
    LOOP_UNLOOPED,
    LOOP_INERTIALIZE_START,
    LOOP_INERTIALIZE_END,
    LOOP_INERTIALIZE_BOTH,
    LOOP_INERTIALIZE_CUBIC,
    LOOP_LINEAR,
    LOOP_LINEAR_INERTIALIZE,
    LOOP_SOFTFADE_INERTIALIZE,
};

static inline void loop_animation(
    slice2d<vec3> looped_bone_positions,
    slice2d<quat> looped_bone_rotations,
    slice2d<vec3> offset_bone_positions,
    slice2d<vec3> offset_bone_rotations,
    const slice2d<vec3> raw_bone_positions,
    const slice2d<quat> raw_bone_rotations,
    const int loop_mode,
    const float halflife_start,
    const float halflife_end,
    const float blendtime_start,
    const float blendtime_end,
    const float ratio,
    const float softfade_duration_start,
    const float softfade_duration_end,
    const float softfade_hardness_start,
    const float softfade_hardness_end,
    const float root_blendtime_start,
    const float root_blendtime_end,
    const bool inertialize_root,
    const float dt)
{    
    array1d<vec3> pos_diff(raw_bone_positions.cols);
    array1d<vec3> vel_diff(raw_bone_positions.cols);
    array1d<vec3> rot_diff(raw_bone_positions.cols);
    array1d<vec3> ang_diff(raw_bone_positions.cols);
    
    switch (loop_mode)
    {
        case LOOP_UNLOOPED: break;
        
        case LOOP_INERTIALIZE_START:
        case LOOP_INERTIALIZE_END:
        case LOOP_INERTIALIZE_BOTH:
        case LOOP_INERTIALIZE_CUBIC:
        case LOOP_LINEAR:
        case LOOP_LINEAR_INERTIALIZE:
            compute_start_end_positional_difference(pos_diff, vel_diff, raw_bone_positions, dt);
            compute_start_end_rotational_difference(rot_diff, ang_diff, raw_bone_rotations, dt);
        break;
        
        case LOOP_SOFTFADE_INERTIALIZE:
            compute_softfade_start_end_difference(
                pos_diff,
                vel_diff,
                raw_bone_positions,
                softfade_duration_start, 
                softfade_duration_end, 
                softfade_hardness_start, 
                softfade_hardness_end, 
                ratio,
                dt);
                
            compute_softfade_start_end_difference(
                rot_diff,
                ang_diff,
                raw_bone_rotations,
                softfade_duration_start, 
                softfade_duration_end, 
                softfade_hardness_start, 
                softfade_hardness_end, 
                ratio,
                dt);
        break;
        
        default: assert(false);
    }
    
    switch (loop_mode)
    {
        case LOOP_UNLOOPED:
            offset_bone_positions.zero();
            offset_bone_rotations.zero();
        break;
        
        case LOOP_INERTIALIZE_START:
            compute_inertialize_start_offsets(offset_bone_positions, pos_diff, vel_diff, halflife_start, dt);
            compute_inertialize_start_offsets(offset_bone_rotations, rot_diff, ang_diff, halflife_start, dt);
        break;
        
        case LOOP_INERTIALIZE_END:
            compute_inertialize_end_offsets(offset_bone_positions, pos_diff, vel_diff, halflife_end, dt);
            compute_inertialize_end_offsets(offset_bone_rotations, rot_diff, ang_diff, halflife_end, dt);
        break;
        
        case LOOP_INERTIALIZE_BOTH:
            compute_inertialize_both_offsets(offset_bone_positions, pos_diff, vel_diff, halflife_start, halflife_end, ratio, dt);
            compute_inertialize_both_offsets(offset_bone_rotations, rot_diff, ang_diff, halflife_start, halflife_end, ratio, dt);
        break;
        
        case LOOP_INERTIALIZE_CUBIC:
            compute_inertialize_cubic_offsets(offset_bone_positions, pos_diff, vel_diff, blendtime_start, blendtime_end, ratio, dt);
            compute_inertialize_cubic_offsets(offset_bone_rotations, rot_diff, ang_diff, blendtime_start, blendtime_end, ratio, dt);
        break;
        
        case LOOP_LINEAR:
            compute_linear_offsets(offset_bone_positions, pos_diff, ratio);
            compute_linear_offsets(offset_bone_rotations, rot_diff, ratio);
        break;
        
        case LOOP_LINEAR_INERTIALIZE:
            compute_linear_inertialize_offsets(offset_bone_positions, pos_diff, vel_diff, blendtime_start, blendtime_end, ratio, dt);
            compute_linear_inertialize_offsets(offset_bone_rotations, rot_diff, ang_diff, blendtime_start, blendtime_end, ratio, dt);
        break;
        
        case LOOP_SOFTFADE_INERTIALIZE:
            compute_softfade_inertialize_offsets(
                offset_bone_positions,
                pos_diff,
                vel_diff,
                blendtime_start, 
                blendtime_end, 
                softfade_duration_start, 
                softfade_duration_end, 
                softfade_hardness_start, 
                softfade_hardness_end, 
                ratio,
                dt);
            compute_softfade_inertialize_offsets(
                offset_bone_rotations,
                rot_diff,
                ang_diff,
                blendtime_start, 
                blendtime_end, 
                softfade_duration_start, 
                softfade_duration_end, 
                softfade_hardness_start, 
                softfade_hardness_end, 
                ratio,
                dt);
        break;

        default: assert(false);
    }
    
    if (inertialize_root)
    {
        compute_root_inertialize_offsets(
            offset_bone_positions,
            offset_bone_rotations,
            raw_bone_positions,
            raw_bone_rotations,
            root_blendtime_start,
            root_blendtime_end,
            ratio,
            dt);
    }
    else
    {
        for (int i = 0; i < offset_bone_positions.rows; i++)
        {
            offset_bone_positions(i, 0) = vec3();
            offset_bone_rotations(i, 0) = vec3();
        }
    }
    
    apply_positional_offsets(looped_bone_positions, raw_bone_positions, offset_bone_positions);
    apply_rotational_offsets(looped_bone_rotations, raw_bone_rotations, offset_bone_rotations);
}

enum
{
    ROOT_FIX = 0,
    ROOT_ACCUMULATE = 1,
    ROOT_LOOP = 2
};

struct anim_clip
{
    const char* name;
    int start, stop;
};

int main(void)
{
    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [animation looping]");
    SetTargetFPS(60);
    
    GuiSetStyle(DEFAULT, TEXT_SIZE, GuiGetStyle(DEFAULT, TEXT_SIZE));    

    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 2.0f, 3.0f, 5.0f };
    camera.target = (Vector3){ -0.5f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    // Ground Plane
    
    Shader ground_plane_shader = LoadShader("./resources/checkerboard.vs", "./resources/checkerboard.fs");
    Mesh ground_plane_mesh = GenMeshPlane(20.0f, 20.0f, 10, 10);
    Model ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;
    
    // Character
    
    character character_data;
    character_load(character_data, "./resources/character.bin");
    
    Shader character_shader = LoadShader("./resources/character.vs", "./resources/character.fs");
    Mesh character_mesh = make_character_mesh(character_data);
    Model character_model = LoadModelFromMesh(character_mesh);
    character_model.materials[0].shader = character_shader;
    
    // Load Animation Data
    
    database db;
    database_load(db, "./resources/database.bin");
    
    // Clips
    
    std::vector<anim_clip> clips =
    {
        { "Short Run Cycle",       358,   394 },
        { "Long Run Cycle",        624,   698 },
        { "Plant and Turn Run",    697,   768 },
        { "Arc Turn Run",         3253,  3337 },
        { "S Turn Run",           3419,  3543 },
        { "Run to Strafe",        4765,  4850 },
        { "Sidestepping Run",     5608,  5671 },
        { "Short Walk Cycle",    13158, 13210 },
        { "90 Degree Turn Walk", 13362, 13456 },
        { "Plant and Turn Walk", 14464, 14565 },
        { "Arc Turn Walk",       15284, 15336 },
        { "Dance 1",             30198, 30245 },
        { "Dance 2",             30301, 30403 },
        { "Dance 3",             30957, 31157 },
        { "Dance 4",             32995, 33052 },
        { "Punch",               40183, 40292 },
        { "Kick",                40454, 40525 },
        { "Uppercut",            41401, 41510 },
    };
    
    std::string clip_combo = "";
    for (int i = 0; i < clips.size(); i++)
    {
        clip_combo += clips[i].name;
        if (i != (int)clips.size() - 1)
        {
            clip_combo += ";";
        }
    }
    
    bool clip_edit = false;
    int clip_index = 0;

    int start_frame = clips[0].start;
    int stop_frame = clips[0].stop; 
    
    // Root Motion
    
    int root_motion = ROOT_FIX;
    bool root_motion_edit = false;
    vec3 root_position = vec3();
    quat root_rotation = quat();
    
    // Looping
    
    int loop_mode = LOOP_UNLOOPED;
    bool loop_mode_edit = false;
    float halflife_start = 0.1f;
    float halflife_end = 0.1f;
    float blendtime_start = 0.2f;
    float blendtime_end = 0.2f;
    float ratio = 0.5f;
    float softfade_duration_start = 0.4f;
    float softfade_duration_end = 0.4f;
    float softfade_hardness_start = 16.0f;
    float softfade_hardness_end = 16.0f;
    float root_blendtime_start = 0.5f;
    float root_blendtime_end = 0.5f;
    bool inertialize_root = true;
  
    // Playback
  
    const float dt = 1.0f / 60.0f;
    float time = 0.0f;
    float playrate = 1.0f;
    bool paused = false;
    
    // Pose Data
    
    array2d<vec3> raw_bone_positions;
    array2d<quat> raw_bone_rotations;
    array2d<vec3> looped_bone_positions;
    array2d<quat> looped_bone_rotations;
    array2d<vec3> offset_bone_positions;
    array2d<vec3> offset_bone_rotations;
    
    raw_bone_positions = db.bone_positions.slice(start_frame, stop_frame);
    raw_bone_rotations = db.bone_rotations.slice(start_frame, stop_frame);
    
    looped_bone_positions = db.bone_positions.slice(start_frame, stop_frame);
    looped_bone_rotations = db.bone_rotations.slice(start_frame, stop_frame);
    
    offset_bone_positions.resize(looped_bone_positions.rows, looped_bone_positions.cols);
    offset_bone_rotations.resize(looped_bone_rotations.rows, looped_bone_rotations.cols);
    
    array1d<vec3> sampled_bone_positions(raw_bone_positions.cols);
    array1d<quat> sampled_bone_rotations(raw_bone_rotations.cols);
    
    animation_sample(sampled_bone_positions, looped_bone_positions, time, dt);
    animation_sample(sampled_bone_rotations, looped_bone_rotations, time, dt);
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    array1d<bool> global_bone_computed(db.nbones());
    
    // Go
    
    auto update_func = [&]()
    {
        // Update Camera
        
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            vec3(0, 0.5f, 0),
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().x : 0.0f,
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().y : 0.0f,
            dt);
        
        // Tick
        
        if (!paused)
        {
            time = fmod(time + playrate * dt, (raw_bone_positions.rows - 1) * dt);            
        }
        
        // Loop Clip
        
        raw_bone_positions = db.bone_positions.slice(start_frame, stop_frame);
        raw_bone_rotations = db.bone_rotations.slice(start_frame, stop_frame);
      
        looped_bone_positions.resize(raw_bone_positions.rows, raw_bone_positions.cols);
        looped_bone_rotations.resize(raw_bone_rotations.rows, raw_bone_rotations.cols);
      
        offset_bone_positions.resize(raw_bone_positions.rows, raw_bone_positions.cols);
        offset_bone_rotations.resize(raw_bone_rotations.rows, raw_bone_rotations.cols);
      
        loop_animation(
            looped_bone_positions,
            looped_bone_rotations,
            offset_bone_positions,
            offset_bone_rotations,
            raw_bone_positions,
            raw_bone_rotations,
            loop_mode,
            halflife_start,
            halflife_end,
            blendtime_start,
            blendtime_end,
            ratio,
            softfade_duration_start,
            softfade_duration_end,
            softfade_hardness_start,
            softfade_hardness_end,
            root_blendtime_start,
            root_blendtime_end,
            inertialize_root,
            dt);
            
        // Sample Animation
        
        animation_sample(sampled_bone_positions, looped_bone_positions, time, dt);
        animation_sample(sampled_bone_rotations, looped_bone_rotations, time, dt);
        
        // Root Motion
        
        if (root_motion == ROOT_FIX)
        {
            sampled_bone_positions(0) = vec3();
            sampled_bone_rotations(0) = quat();
            root_position = vec3();
            root_rotation = quat();
        }
        else if (root_motion == ROOT_LOOP)
        {
            sampled_bone_positions(0) = quat_inv_mul_vec3(looped_bone_rotations(0,0), sampled_bone_positions(0) - looped_bone_positions(0,0));
            sampled_bone_rotations(0) = quat_inv_mul(looped_bone_rotations(0,0), sampled_bone_rotations(0));
            root_position = vec3();
            root_rotation = quat();
        }
        else if (root_motion == ROOT_ACCUMULATE)
        {
            vec3 sampled_root_velocity, sampled_root_angular_velocity;
            animation_sample_root_local_velocity(
                sampled_root_velocity, 
                sampled_root_angular_velocity,
                looped_bone_positions, 
                looped_bone_rotations,
                time, 
                dt);
                
            root_position = quat_mul_vec3(root_rotation, playrate * dt * sampled_root_velocity) + root_position;
            root_rotation = quat_mul(quat_from_scaled_angle_axis(quat_mul_vec3(root_rotation, sampled_root_angular_velocity) * playrate * dt), root_rotation);
            
            sampled_bone_positions(0) = root_position;
            sampled_bone_rotations(0) = root_rotation;
        }
        
        // Done!
        
        forward_kinematics_full(
            global_bone_positions,
            global_bone_rotations,
            sampled_bone_positions,
            sampled_bone_rotations,
            db.bone_parents);

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
        deform_character_mesh(
            character_mesh, 
            character_data, 
            global_bone_positions, 
            global_bone_rotations,
            db.bone_parents);
        
        DrawModel(character_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
        
        draw_axis(global_bone_positions(0), global_bone_rotations(0), 0.25f);
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        EndMode3D();

        // UI
        
        //---------
        
        float ui_hei_plot = 470;
        
        DrawRectangle( 20, ui_hei_plot, 1240, 120, Fade(RAYWHITE, 0.5f));
        DrawRectangleLines( 20, ui_hei_plot, 1240, 120, GRAY);
        DrawLine(20 + 1240/2, ui_hei_plot, 20 + 1240/2, ui_hei_plot + 120, GRAY);

        DrawRectangle( 20, ui_hei_plot + 130, 1240, 60, Fade(RAYWHITE, 0.5f));
        DrawRectangleLines( 20, ui_hei_plot + 130, 1240, 60, GRAY);
        DrawLine(20 + 1240/2, ui_hei_plot + 130, 20 + 1240/2, ui_hei_plot + 130 + 60, GRAY);
        DrawLine(20, ui_hei_plot + 130 + 30, 20 + 1240, ui_hei_plot + 130 + 30, Fade(GRAY, 0.5f));

        float plot_min = FLT_MAX;
        float plot_max = FLT_MIN;
        float offset_plot_min = FLT_MAX;
        float offset_plot_max = FLT_MIN;
        
        for (int i = 0; i < looped_bone_positions.rows; i++)
        {
            plot_min = minf(plot_min, raw_bone_positions(i, 1).x);
            plot_max = maxf(plot_max, raw_bone_positions(i, 1).x);
            plot_min = minf(plot_min, looped_bone_positions(i, 1).x);
            plot_max = maxf(plot_max, looped_bone_positions(i, 1).x);
            offset_plot_min = minf(offset_plot_min, offset_bone_positions(i, 1).x);
            offset_plot_max = maxf(offset_plot_max, offset_bone_positions(i, 1).x);
        }
        
        float plot_scale = (plot_max - plot_min) + 0.01f;
        float plot_center = plot_min + (plot_max - plot_min) / 2.0f;
        float offset_scale = 2.0f * plot_scale;
        
        for (int r = 0; r < 2; r++)
        {
            float time_ratio = time / ((looped_bone_positions.rows - 1) * dt);
            
            DrawLine(20 + time_ratio * 1240/2, ui_hei_plot, 20 + time_ratio * 1240/2, ui_hei_plot + 120, PURPLE);
            DrawLine(20 + 1240/2 + time_ratio/2 * 1240, ui_hei_plot, 20 + 1240/2 + time_ratio * 1240/2, ui_hei_plot + 120, PURPLE);

            for (int i = 0; i < looped_bone_positions.rows - 1; i++)
            {
                float offset0 = (((float)i + 0) / (looped_bone_positions.rows - 1)) * 1240/2 + 20 + r * 1240/2;
                float offset1 = (((float)i + 1) / (looped_bone_positions.rows - 1)) * 1240/2 + 20 + r * 1240/2;
                float raw_value0 = ((raw_bone_positions(i+0, 1).x - plot_center) / plot_scale) * 120 + ui_hei_plot + 60;
                float raw_value1 = ((raw_bone_positions(i+1, 1).x - plot_center) / plot_scale) * 120 + ui_hei_plot + 60;
                
                float loop_value0 = ((looped_bone_positions(i+0, 1).x - plot_center) / plot_scale) * 120 + ui_hei_plot + 60;
                float loop_value1 = ((looped_bone_positions(i+1, 1).x - plot_center) / plot_scale) * 120 + ui_hei_plot + 60;
                
                DrawLine(offset0, loop_value0, offset1, loop_value1, RED);
                DrawLine(offset0, raw_value0, offset1, raw_value1, Fade(RED, 0.5f));
            }
            
            for (int i = 0; i < offset_bone_positions.rows - 1; i++)
            {
                float offset0 = (((float)i + 0) / (offset_bone_positions.rows - 1)) * 1240/2 + 20 + r * 1240/2;
                float offset1 = (((float)i + 1) / (offset_bone_positions.rows - 1)) * 1240/2 + 20 + r * 1240/2;
                float value0 = (offset_bone_positions(i+0, 1).x / offset_scale) * 120 + ui_hei_plot + 130 + 30;
                float value1 = (offset_bone_positions(i+1, 1).x / offset_scale) * 120 + ui_hei_plot + 130 + 30;
                
                DrawLine(offset0, value0, offset1, value1, GREEN);
            }
        }
        
        //---------
        
        float ui_ctrl_hei = 20;
        
        GuiGroupBox((Rectangle){ 1010, ui_ctrl_hei, 250, 60 }, "controls");
        
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  10, 200, 20 }, "Ctrl + Left Click - Move Camera");
        GuiLabel((Rectangle){ 1030, ui_ctrl_hei +  30, 200, 20 }, "Mouse Wheel - Zoom");
        
        //---------
        
        float ui_playback_hei = 90;
        
        GuiGroupBox((Rectangle){ 960, ui_playback_hei, 300, 120 }, "playback");
        
        if (GuiButton((Rectangle){ 1020, ui_playback_hei +  10, 140, 20 }, paused ? "play" : "pause"))
        {
            paused = !paused;
        }
        
        playrate = GuiSliderBar(
            (Rectangle){ 1020, ui_playback_hei +  40, 140, 20 }, 
            "Playrate",
            TextFormat("%7.2f", playrate),
            playrate,
            0.0f,
            2.0f);
        
        if (GuiButton((Rectangle){ 1200, ui_playback_hei +  40, 40, 20 }, "reset"))
        {
            playrate = 1.0f;
        }

        GuiLabel((Rectangle){ 1020, ui_playback_hei +  70, 200, 20 }, "Root Motion");

        if (GuiDropdownBox(
            (Rectangle){ 1020, ui_playback_hei + 90, 140, 20 }, 
            "Fix;Accumulate;Loop",
            &root_motion,
            root_motion_edit))
        {
            root_motion_edit = !root_motion_edit;
        }
        
        if (GuiButton((Rectangle){ 1200, ui_playback_hei + 90, 40, 20 }, "reset"))
        {
            root_position = vec3();
            root_rotation = quat();
        }
        
        //---------
        
        GuiGroupBox((Rectangle){ 350, 20, 160, 40 }, "Clip");

        if (GuiDropdownBox(
            (Rectangle){ 360, 30, 140, 20 }, 
            clip_combo.c_str(),
            &clip_index,
            clip_edit))
        {
            clip_edit = !clip_edit;
            start_frame = clips[clip_index].start;
            stop_frame = clips[clip_index].stop;
            time = 0.0f;
        }
        
        //---------
        
        GuiGroupBox((Rectangle){ 20, 20, 310, 400 }, "Looping Methods");
        
        halflife_start = GuiSliderBar(
            (Rectangle){ 170, 30, 120, 20 }, 
            "halflife start", 
            TextFormat("%5.3f", halflife_start), 
            halflife_start, 0.0f, 0.2f);
            
        halflife_end = GuiSliderBar(
            (Rectangle){ 170, 60, 120, 20 }, 
            "halflife end", 
            TextFormat("%5.3f", halflife_end), 
            halflife_end, 0.0f, 0.2f);
        
        blendtime_start = GuiSliderBar(
            (Rectangle){ 170, 90, 120, 20 }, 
            "blend time start", 
            TextFormat("%5.3f", blendtime_start), 
            blendtime_start, 0.01f, 1.0f);
            
        blendtime_end = GuiSliderBar(
            (Rectangle){ 170, 120, 120, 20 }, 
            "blend time end", 
            TextFormat("%5.3f", blendtime_end), 
            blendtime_end, 0.01f, 1.0f);
        
        ratio = GuiSliderBar(
            (Rectangle){ 170, 150, 120, 20 }, 
            "ratio", 
            TextFormat("%5.3f", ratio), 
            ratio, 0.0f, 1.0f);
        
        softfade_duration_start = GuiSliderBar(
            (Rectangle){ 170, 180, 120, 20 }, 
            "softfade duration start", 
            TextFormat("%5.3f", softfade_duration_start), 
            softfade_duration_start, 0.01f, 1.0f);
            
        softfade_duration_end = GuiSliderBar(
            (Rectangle){ 170, 210, 120, 20 }, 
            "softfade duration end", 
            TextFormat("%5.3f", softfade_duration_end), 
            softfade_duration_end, 0.01f, 1.0f);
        
        softfade_hardness_start = GuiSliderBar(
            (Rectangle){ 170, 240, 120, 20 }, 
            "softfade hardness start", 
            TextFormat("%5.3f", softfade_hardness_start), 
            softfade_hardness_start, 1.0f, 50.0f);
        
        softfade_hardness_end = GuiSliderBar(
            (Rectangle){ 170, 270, 120, 20 }, 
            "softfade hardness end", 
            TextFormat("%5.3f", softfade_hardness_end), 
            softfade_hardness_end, 1.0f, 50.0f);
        
        inertialize_root = GuiCheckBox(
            (Rectangle){ 170, 300, 20, 20 }, 
            "inertialize root",
            inertialize_root);
        
        root_blendtime_start = GuiSliderBar(
            (Rectangle){ 170, 330, 120, 20 }, 
            "root blend time start", 
            TextFormat("%5.3f", root_blendtime_start), 
            root_blendtime_start, 0.01f, 1.0f);
            
        root_blendtime_end = GuiSliderBar(
            (Rectangle){ 170, 360, 120, 20 }, 
            "root blend time end", 
            TextFormat("%5.3f", root_blendtime_end), 
            root_blendtime_end, 0.01f, 1.0f);
        
        if (GuiDropdownBox(
            (Rectangle){ 170, 390, 120, 20 }, 
            "Unlooped;Inertialize Start;Inertialize End;"
            "Inertialize Both;Inertialize Cubic;Linear;Linear Inertialize;"
            "Softfade Inertialize",
            &loop_mode,
            loop_mode_edit))
        {
            loop_mode_edit = !loop_mode_edit;
        }
        
        //---------
        
        float ui_hei_anim = 670;
        
        GuiGroupBox((Rectangle){ 20, ui_hei_anim, 1240, 40 }, "animation");

        time = GuiSliderBar(
            (Rectangle){ 80, ui_hei_anim + 10, 1100, 20 }, 
            "time", 
            TextFormat("%5.3f (%i)", time, start_frame + (int)(time / dt)),
            time,
            0.0f, (looped_bone_positions.rows - 1) * dt);
        
        //---------
              
        EndDrawing();
        
    };
    
#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_func();
    }
#endif

    // Unload stuff and finish
    UnloadModel(character_model);
    UnloadModel(ground_plane_model);
    UnloadShader(character_shader);
    UnloadShader(ground_plane_shader);

    CloseWindow();

    return 0;
}