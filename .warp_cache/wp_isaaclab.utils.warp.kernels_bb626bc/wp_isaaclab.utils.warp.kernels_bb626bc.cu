
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

extern "C" {
}

// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:327
static CUDA_CALLABLE wp::vec_t<3, wp::float32> cast_force_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::quat_t<wp::float32> var_link_quat,
    bool var_is_global)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // forward
    // def cast_force_to_link_frame(force: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:       <L 328>
    // if is_global:                                                                          <L 338>
    if (var_is_global) {
        // return wp.quat_rotate_inv(link_quat, force)                                        <L 339>
        var_0 = wp::quat_rotate_inv(var_link_quat, var_force);
        return var_0;
    }
    if (!var_is_global) {
        // return force                                                                       <L 341>
        return var_force;
    }
    return {};
}


// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:327
static CUDA_CALLABLE void adj_cast_force_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::quat_t<wp::float32> var_link_quat,
    bool var_is_global,
    wp::vec_t<3, wp::float32> & adj_force,
    wp::quat_t<wp::float32> & adj_link_quat,
    bool & adj_is_global,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    //---------
    // forward
    // def cast_force_to_link_frame(force: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:       <L 328>
    // if is_global:                                                                          <L 338>
    if (var_is_global) {
        // return wp.quat_rotate_inv(link_quat, force)                                        <L 339>
        var_0 = wp::quat_rotate_inv(var_link_quat, var_force);
        goto label0;
    }
    if (!var_is_global) {
        // return force                                                                       <L 341>
        goto label1;
    }
    //---------
    // reverse
    if (!var_is_global) {
        label1:;
        adj_force += adj_ret;
        // adj: return force                                                                  <L 341>
    }
    if (var_is_global) {
        label0:;
        adj_0 += adj_ret;
        wp::adj_quat_rotate_inv(var_link_quat, var_force, adj_link_quat, adj_force, adj_0);
        // adj: return wp.quat_rotate_inv(link_quat, force)                                   <L 339>
    }
    // adj: if is_global:                                                                     <L 338>
    // adj: def cast_force_to_link_frame(force: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:  <L 328>
    return;
}


// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:309
static CUDA_CALLABLE wp::vec_t<3, wp::float32> cast_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_position,
    wp::vec_t<3, wp::float32> var_link_position,
    bool var_is_global)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // forward
    // def cast_to_link_frame(position: wp.vec3f, link_position: wp.vec3f, is_global: bool) -> wp.vec3f:       <L 310>
    // if is_global:                                                                          <L 321>
    if (var_is_global) {
        // return position - link_position                                                    <L 322>
        var_0 = wp::sub(var_position, var_link_position);
        return var_0;
    }
    if (!var_is_global) {
        // return position                                                                    <L 324>
        return var_position;
    }
    return {};
}


// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:309
static CUDA_CALLABLE void adj_cast_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_position,
    wp::vec_t<3, wp::float32> var_link_position,
    bool var_is_global,
    wp::vec_t<3, wp::float32> & adj_position,
    wp::vec_t<3, wp::float32> & adj_link_position,
    bool & adj_is_global,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    //---------
    // forward
    // def cast_to_link_frame(position: wp.vec3f, link_position: wp.vec3f, is_global: bool) -> wp.vec3f:       <L 310>
    // if is_global:                                                                          <L 321>
    if (var_is_global) {
        // return position - link_position                                                    <L 322>
        var_0 = wp::sub(var_position, var_link_position);
        goto label0;
    }
    if (!var_is_global) {
        // return position                                                                    <L 324>
        goto label1;
    }
    //---------
    // reverse
    if (!var_is_global) {
        label1:;
        adj_position += adj_ret;
        // adj: return position                                                               <L 324>
    }
    if (var_is_global) {
        label0:;
        adj_0 += adj_ret;
        wp::adj_sub(var_position, var_link_position, adj_position, adj_link_position, adj_0);
        // adj: return position - link_position                                               <L 322>
    }
    // adj: if is_global:                                                                     <L 321>
    // adj: def cast_to_link_frame(position: wp.vec3f, link_position: wp.vec3f, is_global: bool) -> wp.vec3f:  <L 310>
    return;
}


// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:344
static CUDA_CALLABLE wp::vec_t<3, wp::float32> cast_torque_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_torque,
    wp::quat_t<wp::float32> var_link_quat,
    bool var_is_global)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // forward
    // def cast_torque_to_link_frame(torque: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:       <L 345>
    // if is_global:                                                                          <L 356>
    if (var_is_global) {
        // return wp.quat_rotate_inv(link_quat, torque)                                       <L 357>
        var_0 = wp::quat_rotate_inv(var_link_quat, var_torque);
        return var_0;
    }
    if (!var_is_global) {
        // return torque                                                                      <L 359>
        return var_torque;
    }
    return {};
}


// f:/eai/isaaclab/isaaclab/source/isaaclab/isaaclab/utils/warp/kernels.py:344
static CUDA_CALLABLE void adj_cast_torque_to_link_frame_0(
    wp::vec_t<3, wp::float32> var_torque,
    wp::quat_t<wp::float32> var_link_quat,
    bool var_is_global,
    wp::vec_t<3, wp::float32> & adj_torque,
    wp::quat_t<wp::float32> & adj_link_quat,
    bool & adj_is_global,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    //---------
    // forward
    // def cast_torque_to_link_frame(torque: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:       <L 345>
    // if is_global:                                                                          <L 356>
    if (var_is_global) {
        // return wp.quat_rotate_inv(link_quat, torque)                                       <L 357>
        var_0 = wp::quat_rotate_inv(var_link_quat, var_torque);
        goto label0;
    }
    if (!var_is_global) {
        // return torque                                                                      <L 359>
        goto label1;
    }
    //---------
    // reverse
    if (!var_is_global) {
        label1:;
        adj_torque += adj_ret;
        // adj: return torque                                                                 <L 359>
    }
    if (var_is_global) {
        label0:;
        adj_0 += adj_ret;
        wp::adj_quat_rotate_inv(var_link_quat, var_torque, adj_link_quat, adj_torque, adj_0);
        // adj: return wp.quat_rotate_inv(link_quat, torque)                                  <L 357>
    }
    // adj: if is_global:                                                                     <L 356>
    // adj: def cast_torque_to_link_frame(torque: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:  <L 345>
    return;
}



extern "C" __global__ void raycast_mesh_kernel_2b491393_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::uint64 var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::float32 var_max_dist,
    wp::int32 var_return_distance,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        bool var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        const wp::int32 var_23 = 1;
        bool var_24;
        const wp::int32 var_25 = 1;
        bool var_26;
        const wp::int32 var_27 = 1;
        bool var_28;
        //---------
        // forward
        // def raycast_mesh_kernel(                                                               <L 18>
        // tid = wp.tid()                                                                         <L 60>
        var_0 = builtin_tid1d();
        // t = float(0.0)  # hit distance along ray                                               <L 62>
        var_2 = wp::float(var_1);
        // u = float(0.0)  # hit face barycentric u                                               <L 63>
        var_4 = wp::float(var_3);
        // v = float(0.0)  # hit face barycentric v                                               <L 64>
        var_6 = wp::float(var_5);
        // sign = float(0.0)  # hit face sign                                                     <L 65>
        var_8 = wp::float(var_7);
        // n = wp.vec3()  # hit face normal                                                       <L 66>
        var_9 = wp::vec_t<3, wp::float32>();
        // f = int(0)  # hit face index                                                           <L 67>
        var_11 = wp::int(var_10);
        // hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)       <L 70>
        var_12 = wp::address(var_ray_starts, var_0);
        var_13 = wp::address(var_ray_directions, var_0);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = wp::mesh_query_ray(var_mesh, var_15, var_16, var_max_dist, var_2, var_4, var_6, var_8, var_9, var_11);
        // if hit_success:                                                                        <L 72>
        if (var_14) {
            // ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]                          <L 73>
            var_17 = wp::address(var_ray_starts, var_0);
            var_18 = wp::address(var_ray_directions, var_0);
            var_20 = wp::load(var_18);
            var_19 = wp::mul(var_2, var_20);
            var_22 = wp::load(var_17);
            var_21 = wp::add(var_22, var_19);
            wp::array_store(var_ray_hits, var_0, var_21);
            // if return_distance == 1:                                                           <L 74>
            var_24 = (var_return_distance == var_23);
            if (var_24) {
                // ray_distance[tid] = t                                                          <L 75>
                wp::array_store(var_ray_distance, var_0, var_2);
            }
            // if return_normal == 1:                                                             <L 76>
            var_26 = (var_return_normal == var_25);
            if (var_26) {
                // ray_normal[tid] = n                                                            <L 77>
                wp::array_store(var_ray_normal, var_0, var_9);
            }
            // if return_face_id == 1:                                                            <L 78>
            var_28 = (var_return_face_id == var_27);
            if (var_28) {
                // ray_face_id[tid] = f                                                           <L 79>
                wp::array_store(var_ray_face_id, var_0, var_11);
            }
        }
    }
}



extern "C" __global__ void raycast_dynamic_meshes_kernel_847eae7a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint64> var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::array_t<wp::int16> var_ray_mesh_id,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_mesh_rotations,
    wp::float32 var_max_dist,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id,
    wp::int32 var_return_mesh_id)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::quat_t<wp::float32>* var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::uint64* var_15;
        wp::mesh_query_ray_t var_16;
        wp::uint64 var_17;
        bool* var_18;
        bool var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::float32* var_24;
        bool var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::float32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        const wp::int32 var_33 = 1;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::int32 var_38 = 1;
        bool var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        bool var_43;
        wp::int16 var_44;
        bool var_45;
        //---------
        // forward
        // def raycast_dynamic_meshes_kernel(                                                     <L 162>
        // tid_mesh_id, tid_env, tid_ray = wp.tid()                                               <L 216>
        builtin_tid3d(var_0, var_1, var_2);
        // mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])       <L 218>
        var_3 = wp::address(var_mesh_positions, var_1, var_0);
        var_4 = wp::address(var_mesh_rotations, var_1, var_0);
        var_6 = wp::load(var_3);
        var_7 = wp::load(var_4);
        var_5 = wp::transform_t<wp::float32>(var_6, var_7);
        // mesh_pose_inv = wp.transform_inverse(mesh_pose)                                        <L 219>
        var_8 = wp::transform_inverse(var_5);
        // direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])       <L 220>
        var_9 = wp::address(var_ray_directions, var_1, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::transform_vector(var_8, var_11);
        // start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])            <L 221>
        var_12 = wp::address(var_ray_starts, var_1, var_2);
        var_14 = wp::load(var_12);
        var_13 = wp::transform_point(var_8, var_14);
        // mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)       <L 224>
        var_15 = wp::address(var_mesh, var_1, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::mesh_query_ray(var_17, var_13, var_10, var_max_dist);
        // if mesh_query_ray_t.result:                                                            <L 226>
        var_18 = &(var_16.result);
        var_19 = wp::load(var_18);
        if (var_19) {
            // wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)                  <L 227>
            var_20 = &(var_16.t);
            var_22 = wp::load(var_20);
            var_21 = wp::atomic_min(var_ray_distance, var_1, var_2, var_22);
            // if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:                           <L 232>
            var_23 = &(var_16.t);
            var_24 = wp::address(var_ray_distance, var_1, var_2);
            var_26 = wp::load(var_23);
            var_27 = wp::load(var_24);
            var_25 = (var_26 == var_27);
            if (var_25) {
                // hit_pos = start_pos + mesh_query_ray_t.t * direction                           <L 234>
                var_28 = &(var_16.t);
                var_30 = wp::load(var_28);
                var_29 = wp::mul(var_30, var_10);
                var_31 = wp::add(var_13, var_29);
                // ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)            <L 235>
                var_32 = wp::transform_point(var_5, var_31);
                wp::array_store(var_ray_hits, var_1, var_2, var_32);
                // if return_normal == 1:                                                         <L 238>
                var_34 = (var_return_normal == var_33);
                if (var_34) {
                    // n = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)                <L 239>
                    var_35 = &(var_16.normal);
                    var_37 = wp::load(var_35);
                    var_36 = wp::transform_vector(var_5, var_37);
                    // ray_normal[tid_env, tid_ray] = n                                           <L 240>
                    wp::array_store(var_ray_normal, var_1, var_2, var_36);
                }
                // if return_face_id == 1:                                                        <L 241>
                var_39 = (var_return_face_id == var_38);
                if (var_39) {
                    // ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face                      <L 242>
                    var_40 = &(var_16.face);
                    var_41 = wp::load(var_40);
                    wp::array_store(var_ray_face_id, var_1, var_2, var_41);
                }
                // if return_mesh_id == 1:                                                        <L 243>
                var_43 = (var_return_mesh_id == var_42);
                if (var_43) {
                    // ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)                      <L 244>
                    var_44 = wp::int16(var_0);
                    wp::array_store(var_ray_mesh_id, var_1, var_2, var_44);
                }
            }
        }
        var_45 = wp::load(var_18);
    }
}



extern "C" __global__ void add_forces_and_torques_at_position_4536725f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_torques_b,
    bool var_is_global)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::int32* var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::quat_t<wp::float32>* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::int32* var_36;
        wp::int32* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::int32* var_42;
        wp::int32* var_43;
        wp::quat_t<wp::float32>* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::quat_t<wp::float32> var_49;
        wp::int32* var_50;
        wp::int32* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        //---------
        // forward
        // def add_forces_and_torques_at_position(                                                <L 363>
        // tid_env, tid_body = wp.tid()                                                           <L 393>
        builtin_tid2d(var_0, var_1);
        // if forces:                                                                             <L 396>
        if (var_forces) {
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(       <L 398>
            // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 399>
            var_2 = wp::address(var_forces, var_0, var_1);
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_link_quaternions, var_6, var_7);
            var_9 = wp::load(var_2);
            var_10 = wp::load(var_5);
            var_8 = cast_force_to_link_frame_0(var_9, var_10, var_is_global);
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(       <L 398>
            var_11 = wp::address(var_env_ids, var_0);
            var_12 = wp::address(var_body_ids, var_1);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            var_13 = wp::atomic_add(var_composed_forces_b, var_14, var_15, var_8);
            // if positions:                                                                      <L 402>
            if (var_positions) {
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(           <L 403>
                // cast_to_link_frame(                                                            <L 404>
                // positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 405>
                var_16 = wp::address(var_positions, var_0, var_1);
                var_17 = wp::address(var_env_ids, var_0);
                var_18 = wp::address(var_body_ids, var_1);
                var_20 = wp::load(var_17);
                var_21 = wp::load(var_18);
                var_19 = wp::address(var_link_positions, var_20, var_21);
                var_23 = wp::load(var_16);
                var_24 = wp::load(var_19);
                var_22 = cast_to_link_frame_0(var_23, var_24, var_is_global);
                var_25 = wp::skew(var_22);
                // ) @ cast_force_to_link_frame(                                                  <L 407>
                // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 408>
                var_26 = wp::address(var_forces, var_0, var_1);
                var_27 = wp::address(var_env_ids, var_0);
                var_28 = wp::address(var_body_ids, var_1);
                var_30 = wp::load(var_27);
                var_31 = wp::load(var_28);
                var_29 = wp::address(var_link_quaternions, var_30, var_31);
                var_33 = wp::load(var_26);
                var_34 = wp::load(var_29);
                var_32 = cast_force_to_link_frame_0(var_33, var_34, var_is_global);
                var_35 = wp::mul(var_25, var_32);
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(           <L 403>
                var_36 = wp::address(var_env_ids, var_0);
                var_37 = wp::address(var_body_ids, var_1);
                var_39 = wp::load(var_36);
                var_40 = wp::load(var_37);
                var_38 = wp::atomic_add(var_composed_torques_b, var_39, var_40, var_35);
            }
        }
        // if torques:                                                                            <L 410>
        if (var_torques) {
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(       <L 411>
            // torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 412>
            var_41 = wp::address(var_torques, var_0, var_1);
            var_42 = wp::address(var_env_ids, var_0);
            var_43 = wp::address(var_body_ids, var_1);
            var_45 = wp::load(var_42);
            var_46 = wp::load(var_43);
            var_44 = wp::address(var_link_quaternions, var_45, var_46);
            var_48 = wp::load(var_41);
            var_49 = wp::load(var_44);
            var_47 = cast_torque_to_link_frame_0(var_48, var_49, var_is_global);
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(       <L 411>
            var_50 = wp::address(var_env_ids, var_0);
            var_51 = wp::address(var_body_ids, var_1);
            var_53 = wp::load(var_50);
            var_54 = wp::load(var_51);
            var_52 = wp::atomic_add(var_composed_torques_b, var_53, var_54, var_47);
        }
    }
}



extern "C" __global__ void add_forces_and_torques_at_position_4536725f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_torques_b,
    bool var_is_global,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> adj_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_composed_torques_b,
    bool adj_is_global)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::int32* var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::quat_t<wp::float32>* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::int32* var_36;
        wp::int32* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::int32* var_42;
        wp::int32* var_43;
        wp::quat_t<wp::float32>* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::quat_t<wp::float32> var_49;
        wp::int32* var_50;
        wp::int32* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::quat_t<wp::float32> adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::mat_t<3, 3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::quat_t<wp::float32> adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::quat_t<wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::quat_t<wp::float32> adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::quat_t<wp::float32> adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        //---------
        // forward
        // def add_forces_and_torques_at_position(                                                <L 363>
        // tid_env, tid_body = wp.tid()                                                           <L 393>
        builtin_tid2d(var_0, var_1);
        // if forces:                                                                             <L 396>
        if (var_forces) {
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(       <L 398>
            // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 399>
            var_2 = wp::address(var_forces, var_0, var_1);
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_link_quaternions, var_6, var_7);
            var_9 = wp::load(var_2);
            var_10 = wp::load(var_5);
            var_8 = cast_force_to_link_frame_0(var_9, var_10, var_is_global);
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(       <L 398>
            var_11 = wp::address(var_env_ids, var_0);
            var_12 = wp::address(var_body_ids, var_1);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            // var_13 = wp::atomic_add(var_composed_forces_b, var_14, var_15, var_8);
            // if positions:                                                                      <L 402>
            if (var_positions) {
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(           <L 403>
                // cast_to_link_frame(                                                            <L 404>
                // positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 405>
                var_16 = wp::address(var_positions, var_0, var_1);
                var_17 = wp::address(var_env_ids, var_0);
                var_18 = wp::address(var_body_ids, var_1);
                var_20 = wp::load(var_17);
                var_21 = wp::load(var_18);
                var_19 = wp::address(var_link_positions, var_20, var_21);
                var_23 = wp::load(var_16);
                var_24 = wp::load(var_19);
                var_22 = cast_to_link_frame_0(var_23, var_24, var_is_global);
                var_25 = wp::skew(var_22);
                // ) @ cast_force_to_link_frame(                                                  <L 407>
                // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 408>
                var_26 = wp::address(var_forces, var_0, var_1);
                var_27 = wp::address(var_env_ids, var_0);
                var_28 = wp::address(var_body_ids, var_1);
                var_30 = wp::load(var_27);
                var_31 = wp::load(var_28);
                var_29 = wp::address(var_link_quaternions, var_30, var_31);
                var_33 = wp::load(var_26);
                var_34 = wp::load(var_29);
                var_32 = cast_force_to_link_frame_0(var_33, var_34, var_is_global);
                var_35 = wp::mul(var_25, var_32);
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(           <L 403>
                var_36 = wp::address(var_env_ids, var_0);
                var_37 = wp::address(var_body_ids, var_1);
                var_39 = wp::load(var_36);
                var_40 = wp::load(var_37);
                // var_38 = wp::atomic_add(var_composed_torques_b, var_39, var_40, var_35);
            }
        }
        // if torques:                                                                            <L 410>
        if (var_torques) {
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(       <L 411>
            // torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 412>
            var_41 = wp::address(var_torques, var_0, var_1);
            var_42 = wp::address(var_env_ids, var_0);
            var_43 = wp::address(var_body_ids, var_1);
            var_45 = wp::load(var_42);
            var_46 = wp::load(var_43);
            var_44 = wp::address(var_link_quaternions, var_45, var_46);
            var_48 = wp::load(var_41);
            var_49 = wp::load(var_44);
            var_47 = cast_torque_to_link_frame_0(var_48, var_49, var_is_global);
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(       <L 411>
            var_50 = wp::address(var_env_ids, var_0);
            var_51 = wp::address(var_body_ids, var_1);
            var_53 = wp::load(var_50);
            var_54 = wp::load(var_51);
            // var_52 = wp::atomic_add(var_composed_torques_b, var_53, var_54, var_47);
        }
        //---------
        // reverse
        if (var_torques) {
            wp::adj_atomic_add(var_composed_torques_b, var_53, var_54, var_47, adj_composed_torques_b, adj_50, adj_51, adj_47, adj_52);
            wp::adj_load(var_51, adj_51, adj_54);
            wp::adj_load(var_50, adj_50, adj_53);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_51);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_50);
            // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(  <L 411>
            adj_cast_torque_to_link_frame_0(var_48, var_49, var_is_global, adj_41, adj_44, adj_is_global, adj_47);
            wp::adj_load(var_44, adj_44, adj_49);
            wp::adj_load(var_41, adj_41, adj_48);
            wp::adj_address(var_link_quaternions, var_45, var_46, adj_link_quaternions, adj_42, adj_43, adj_44);
            wp::adj_load(var_43, adj_43, adj_46);
            wp::adj_load(var_42, adj_42, adj_45);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_43);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_42);
            wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_41);
            // adj: torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 412>
            // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(  <L 411>
        }
        // adj: if torques:                                                                       <L 410>
        if (var_forces) {
            if (var_positions) {
                wp::adj_atomic_add(var_composed_torques_b, var_39, var_40, var_35, adj_composed_torques_b, adj_36, adj_37, adj_35, adj_38);
                wp::adj_load(var_37, adj_37, adj_40);
                wp::adj_load(var_36, adj_36, adj_39);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_37);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_36);
                // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(      <L 403>
                wp::adj_mul(var_25, var_32, adj_25, adj_32, adj_35);
                adj_cast_force_to_link_frame_0(var_33, var_34, var_is_global, adj_26, adj_29, adj_is_global, adj_32);
                wp::adj_load(var_29, adj_29, adj_34);
                wp::adj_load(var_26, adj_26, adj_33);
                wp::adj_address(var_link_quaternions, var_30, var_31, adj_link_quaternions, adj_27, adj_28, adj_29);
                wp::adj_load(var_28, adj_28, adj_31);
                wp::adj_load(var_27, adj_27, adj_30);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_28);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_27);
                wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_26);
                // adj: forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 408>
                // adj: ) @ cast_force_to_link_frame(                                             <L 407>
                wp::adj_skew(var_22, adj_22, adj_25);
                adj_cast_to_link_frame_0(var_23, var_24, var_is_global, adj_16, adj_19, adj_is_global, adj_22);
                wp::adj_load(var_19, adj_19, adj_24);
                wp::adj_load(var_16, adj_16, adj_23);
                wp::adj_address(var_link_positions, var_20, var_21, adj_link_positions, adj_17, adj_18, adj_19);
                wp::adj_load(var_18, adj_18, adj_21);
                wp::adj_load(var_17, adj_17, adj_20);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_18);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_17);
                wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_16);
                // adj: positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 405>
                // adj: cast_to_link_frame(                                                       <L 404>
                // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(      <L 403>
            }
            // adj: if positions:                                                                 <L 402>
            wp::adj_atomic_add(var_composed_forces_b, var_14, var_15, var_8, adj_composed_forces_b, adj_11, adj_12, adj_8, adj_13);
            wp::adj_load(var_12, adj_12, adj_15);
            wp::adj_load(var_11, adj_11, adj_14);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_12);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_11);
            // adj: composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(  <L 398>
            adj_cast_force_to_link_frame_0(var_9, var_10, var_is_global, adj_2, adj_5, adj_is_global, adj_8);
            wp::adj_load(var_5, adj_5, adj_10);
            wp::adj_load(var_2, adj_2, adj_9);
            wp::adj_address(var_link_quaternions, var_6, var_7, adj_link_quaternions, adj_3, adj_4, adj_5);
            wp::adj_load(var_4, adj_4, adj_7);
            wp::adj_load(var_3, adj_3, adj_6);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_4);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_2);
            // adj: forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 399>
            // adj: composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(  <L 398>
        }
        // adj: if forces:                                                                        <L 396>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 393>
        // adj: def add_forces_and_torques_at_position(                                           <L 363>
        continue;
    }
}



extern "C" __global__ void set_forces_and_torques_at_position_c4268548_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_torques_b,
    bool var_is_global)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::quat_t<wp::float32>* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::quat_t<wp::float32>* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::quat_t<wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::int32* var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // forward
        // def set_forces_and_torques_at_position(                                                <L 417>
        // tid_env, tid_body = wp.tid()                                                           <L 447>
        builtin_tid2d(var_0, var_1);
        // if torques:                                                                            <L 450>
        if (var_torques) {
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(       <L 451>
            // torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 452>
            var_2 = wp::address(var_torques, var_0, var_1);
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_link_quaternions, var_6, var_7);
            var_9 = wp::load(var_2);
            var_10 = wp::load(var_5);
            var_8 = cast_torque_to_link_frame_0(var_9, var_10, var_is_global);
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(       <L 451>
            var_11 = wp::address(var_env_ids, var_0);
            var_12 = wp::address(var_body_ids, var_1);
            var_13 = wp::load(var_11);
            var_14 = wp::load(var_12);
            wp::array_store(var_composed_torques_b, var_13, var_14, var_8);
        }
        // if forces:                                                                             <L 456>
        if (var_forces) {
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(       <L 458>
            // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 459>
            var_15 = wp::address(var_forces, var_0, var_1);
            var_16 = wp::address(var_env_ids, var_0);
            var_17 = wp::address(var_body_ids, var_1);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::address(var_link_quaternions, var_19, var_20);
            var_22 = wp::load(var_15);
            var_23 = wp::load(var_18);
            var_21 = cast_force_to_link_frame_0(var_22, var_23, var_is_global);
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(       <L 458>
            var_24 = wp::address(var_env_ids, var_0);
            var_25 = wp::address(var_body_ids, var_1);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_25);
            wp::array_store(var_composed_forces_b, var_26, var_27, var_21);
            // if positions:                                                                      <L 462>
            if (var_positions) {
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(            <L 463>
                // cast_to_link_frame(                                                            <L 464>
                // positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 465>
                var_28 = wp::address(var_positions, var_0, var_1);
                var_29 = wp::address(var_env_ids, var_0);
                var_30 = wp::address(var_body_ids, var_1);
                var_32 = wp::load(var_29);
                var_33 = wp::load(var_30);
                var_31 = wp::address(var_link_positions, var_32, var_33);
                var_35 = wp::load(var_28);
                var_36 = wp::load(var_31);
                var_34 = cast_to_link_frame_0(var_35, var_36, var_is_global);
                var_37 = wp::skew(var_34);
                // ) @ cast_force_to_link_frame(                                                  <L 467>
                // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 468>
                var_38 = wp::address(var_forces, var_0, var_1);
                var_39 = wp::address(var_env_ids, var_0);
                var_40 = wp::address(var_body_ids, var_1);
                var_42 = wp::load(var_39);
                var_43 = wp::load(var_40);
                var_41 = wp::address(var_link_quaternions, var_42, var_43);
                var_45 = wp::load(var_38);
                var_46 = wp::load(var_41);
                var_44 = cast_force_to_link_frame_0(var_45, var_46, var_is_global);
                var_47 = wp::mul(var_37, var_44);
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(            <L 463>
                var_48 = wp::address(var_env_ids, var_0);
                var_49 = wp::address(var_body_ids, var_1);
                var_50 = wp::load(var_48);
                var_51 = wp::load(var_49);
                wp::array_store(var_composed_torques_b, var_50, var_51, var_47);
            }
        }
    }
}



extern "C" __global__ void set_forces_and_torques_at_position_c4268548_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_composed_torques_b,
    bool var_is_global,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_link_positions,
    wp::array_t<wp::quat_t<wp::float32>> adj_link_quaternions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_composed_forces_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_composed_torques_b,
    bool adj_is_global)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::quat_t<wp::float32>* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::quat_t<wp::float32>* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::quat_t<wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::int32* var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::quat_t<wp::float32> adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::quat_t<wp::float32> adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::quat_t<wp::float32> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::mat_t<3, 3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::quat_t<wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::quat_t<wp::float32> adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        //---------
        // forward
        // def set_forces_and_torques_at_position(                                                <L 417>
        // tid_env, tid_body = wp.tid()                                                           <L 447>
        builtin_tid2d(var_0, var_1);
        // if torques:                                                                            <L 450>
        if (var_torques) {
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(       <L 451>
            // torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 452>
            var_2 = wp::address(var_torques, var_0, var_1);
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_link_quaternions, var_6, var_7);
            var_9 = wp::load(var_2);
            var_10 = wp::load(var_5);
            var_8 = cast_torque_to_link_frame_0(var_9, var_10, var_is_global);
            // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(       <L 451>
            var_11 = wp::address(var_env_ids, var_0);
            var_12 = wp::address(var_body_ids, var_1);
            var_13 = wp::load(var_11);
            var_14 = wp::load(var_12);
            // wp::array_store(var_composed_torques_b, var_13, var_14, var_8);
        }
        // if forces:                                                                             <L 456>
        if (var_forces) {
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(       <L 458>
            // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 459>
            var_15 = wp::address(var_forces, var_0, var_1);
            var_16 = wp::address(var_env_ids, var_0);
            var_17 = wp::address(var_body_ids, var_1);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::address(var_link_quaternions, var_19, var_20);
            var_22 = wp::load(var_15);
            var_23 = wp::load(var_18);
            var_21 = cast_force_to_link_frame_0(var_22, var_23, var_is_global);
            // composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(       <L 458>
            var_24 = wp::address(var_env_ids, var_0);
            var_25 = wp::address(var_body_ids, var_1);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_25);
            // wp::array_store(var_composed_forces_b, var_26, var_27, var_21);
            // if positions:                                                                      <L 462>
            if (var_positions) {
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(            <L 463>
                // cast_to_link_frame(                                                            <L 464>
                // positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 465>
                var_28 = wp::address(var_positions, var_0, var_1);
                var_29 = wp::address(var_env_ids, var_0);
                var_30 = wp::address(var_body_ids, var_1);
                var_32 = wp::load(var_29);
                var_33 = wp::load(var_30);
                var_31 = wp::address(var_link_positions, var_32, var_33);
                var_35 = wp::load(var_28);
                var_36 = wp::load(var_31);
                var_34 = cast_to_link_frame_0(var_35, var_36, var_is_global);
                var_37 = wp::skew(var_34);
                // ) @ cast_force_to_link_frame(                                                  <L 467>
                // forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global       <L 468>
                var_38 = wp::address(var_forces, var_0, var_1);
                var_39 = wp::address(var_env_ids, var_0);
                var_40 = wp::address(var_body_ids, var_1);
                var_42 = wp::load(var_39);
                var_43 = wp::load(var_40);
                var_41 = wp::address(var_link_quaternions, var_42, var_43);
                var_45 = wp::load(var_38);
                var_46 = wp::load(var_41);
                var_44 = cast_force_to_link_frame_0(var_45, var_46, var_is_global);
                var_47 = wp::mul(var_37, var_44);
                // composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(            <L 463>
                var_48 = wp::address(var_env_ids, var_0);
                var_49 = wp::address(var_body_ids, var_1);
                var_50 = wp::load(var_48);
                var_51 = wp::load(var_49);
                // wp::array_store(var_composed_torques_b, var_50, var_51, var_47);
            }
        }
        //---------
        // reverse
        if (var_forces) {
            if (var_positions) {
                wp::adj_array_store(var_composed_torques_b, var_50, var_51, var_47, adj_composed_torques_b, adj_48, adj_49, adj_47);
                wp::adj_load(var_49, adj_49, adj_51);
                wp::adj_load(var_48, adj_48, adj_50);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_49);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_48);
                // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(       <L 463>
                wp::adj_mul(var_37, var_44, adj_37, adj_44, adj_47);
                adj_cast_force_to_link_frame_0(var_45, var_46, var_is_global, adj_38, adj_41, adj_is_global, adj_44);
                wp::adj_load(var_41, adj_41, adj_46);
                wp::adj_load(var_38, adj_38, adj_45);
                wp::adj_address(var_link_quaternions, var_42, var_43, adj_link_quaternions, adj_39, adj_40, adj_41);
                wp::adj_load(var_40, adj_40, adj_43);
                wp::adj_load(var_39, adj_39, adj_42);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_40);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_39);
                wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_38);
                // adj: forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 468>
                // adj: ) @ cast_force_to_link_frame(                                             <L 467>
                wp::adj_skew(var_34, adj_34, adj_37);
                adj_cast_to_link_frame_0(var_35, var_36, var_is_global, adj_28, adj_31, adj_is_global, adj_34);
                wp::adj_load(var_31, adj_31, adj_36);
                wp::adj_load(var_28, adj_28, adj_35);
                wp::adj_address(var_link_positions, var_32, var_33, adj_link_positions, adj_29, adj_30, adj_31);
                wp::adj_load(var_30, adj_30, adj_33);
                wp::adj_load(var_29, adj_29, adj_32);
                wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_30);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_29);
                wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_28);
                // adj: positions[tid_env, tid_body], link_positions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 465>
                // adj: cast_to_link_frame(                                                       <L 464>
                // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(       <L 463>
            }
            // adj: if positions:                                                                 <L 462>
            wp::adj_array_store(var_composed_forces_b, var_26, var_27, var_21, adj_composed_forces_b, adj_24, adj_25, adj_21);
            wp::adj_load(var_25, adj_25, adj_27);
            wp::adj_load(var_24, adj_24, adj_26);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_25);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_24);
            // adj: composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(  <L 458>
            adj_cast_force_to_link_frame_0(var_22, var_23, var_is_global, adj_15, adj_18, adj_is_global, adj_21);
            wp::adj_load(var_18, adj_18, adj_23);
            wp::adj_load(var_15, adj_15, adj_22);
            wp::adj_address(var_link_quaternions, var_19, var_20, adj_link_quaternions, adj_16, adj_17, adj_18);
            wp::adj_load(var_17, adj_17, adj_20);
            wp::adj_load(var_16, adj_16, adj_19);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_17);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_16);
            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_15);
            // adj: forces[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 459>
            // adj: composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(  <L 458>
        }
        // adj: if forces:                                                                        <L 456>
        if (var_torques) {
            wp::adj_array_store(var_composed_torques_b, var_13, var_14, var_8, adj_composed_torques_b, adj_11, adj_12, adj_8);
            wp::adj_load(var_12, adj_12, adj_14);
            wp::adj_load(var_11, adj_11, adj_13);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_12);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_11);
            // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(  <L 451>
            adj_cast_torque_to_link_frame_0(var_9, var_10, var_is_global, adj_2, adj_5, adj_is_global, adj_8);
            wp::adj_load(var_5, adj_5, adj_10);
            wp::adj_load(var_2, adj_2, adj_9);
            wp::adj_address(var_link_quaternions, var_6, var_7, adj_link_quaternions, adj_3, adj_4, adj_5);
            wp::adj_load(var_4, adj_4, adj_7);
            wp::adj_load(var_3, adj_3, adj_6);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_4);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
            wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_2);
            // adj: torques[tid_env, tid_body], link_quaternions[env_ids[tid_env], body_ids[tid_body]], is_global  <L 452>
            // adj: composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(  <L 451>
        }
        // adj: if torques:                                                                       <L 450>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 447>
        // adj: def set_forces_and_torques_at_position(                                           <L 417>
        continue;
    }
}



extern "C" __global__ void reshape_tiled_image_5f4bc01e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_tiled_image_buffer,
    wp::array_t<wp::uint32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint32* var_18;
        wp::uint32 var_19;
        wp::uint32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_43f9491f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_tiled_image_buffer,
    wp::array_t<wp::uint8> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint8* var_18;
        wp::uint8 var_19;
        wp::uint8 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint8(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_45364ab5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_tiled_image_buffer,
    wp::array_t<wp::float32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::float32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void raycast_static_meshes_kernel_5fd2a236_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint64> var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::array_t<wp::int16> var_ray_mesh_id,
    wp::float32 var_max_dist,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id,
    wp::int32 var_return_mesh_id)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::uint64* var_9;
        wp::mesh_query_ray_t var_10;
        wp::uint64 var_11;
        bool* var_12;
        bool var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32* var_18;
        bool var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::float32 var_24;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        const wp::int32 var_30 = 1;
        bool var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        bool var_35;
        wp::int16 var_36;
        bool var_37;
        //---------
        // forward
        // def raycast_static_meshes_kernel(                                                      <L 83>
        // tid_mesh_id, tid_env, tid_ray = wp.tid()                                               <L 133>
        builtin_tid3d(var_0, var_1, var_2);
        // direction = ray_directions[tid_env, tid_ray]                                           <L 135>
        var_3 = wp::address(var_ray_directions, var_1, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // start_pos = ray_starts[tid_env, tid_ray]                                               <L 136>
        var_6 = wp::address(var_ray_starts, var_1, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)       <L 139>
        var_9 = wp::address(var_mesh, var_1, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::mesh_query_ray(var_11, var_7, var_4, var_max_dist);
        // if mesh_query_ray_t.result:                                                            <L 142>
        var_12 = &(var_10.result);
        var_13 = wp::load(var_12);
        if (var_13) {
            // wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)                  <L 143>
            var_14 = &(var_10.t);
            var_16 = wp::load(var_14);
            var_15 = wp::atomic_min(var_ray_distance, var_1, var_2, var_16);
            // if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:                           <L 148>
            var_17 = &(var_10.t);
            var_18 = wp::address(var_ray_distance, var_1, var_2);
            var_20 = wp::load(var_17);
            var_21 = wp::load(var_18);
            var_19 = (var_20 == var_21);
            if (var_19) {
                // ray_hits[tid_env, tid_ray] = start_pos + mesh_query_ray_t.t * direction        <L 150>
                var_22 = &(var_10.t);
                var_24 = wp::load(var_22);
                var_23 = wp::mul(var_24, var_4);
                var_25 = wp::add(var_7, var_23);
                wp::array_store(var_ray_hits, var_1, var_2, var_25);
                // if return_normal == 1:                                                         <L 153>
                var_27 = (var_return_normal == var_26);
                if (var_27) {
                    // ray_normal[tid_env, tid_ray] = mesh_query_ray_t.normal                     <L 154>
                    var_28 = &(var_10.normal);
                    var_29 = wp::load(var_28);
                    wp::array_store(var_ray_normal, var_1, var_2, var_29);
                }
                // if return_face_id == 1:                                                        <L 155>
                var_31 = (var_return_face_id == var_30);
                if (var_31) {
                    // ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face                      <L 156>
                    var_32 = &(var_10.face);
                    var_33 = wp::load(var_32);
                    wp::array_store(var_ray_face_id, var_1, var_2, var_33);
                }
                // if return_mesh_id == 1:                                                        <L 157>
                var_35 = (var_return_mesh_id == var_34);
                if (var_35) {
                    // ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)                      <L 158>
                    var_36 = wp::int16(var_0);
                    wp::array_store(var_ray_mesh_id, var_1, var_2, var_36);
                }
            }
        }
        var_37 = wp::load(var_12);
    }
}

