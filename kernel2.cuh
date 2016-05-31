

//To check if the given value lies betwen x1 and x2
__device__ bool InRange(real_d x1,real_d x2, real_d value, const  real_d  *const_args, int dim);

//Reset the buffer with 0s
__global__ SetToZero(real_l *buffer, real_l buffer_size){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < buffer_size){
        buffer[idx] = 0;
    }
}

//To update the linked lists after every iteration using a cell parallel approach. This kernel should be launched using 3d thread blocks
__global__ void updateLists(real_l *cell_list, real_l *particle_list, const real_d *position, const real_d *const_args, const real_l num_particles){
    real_l idx = threadIdx.x+blockIdx.x*blockDim.x;
    real_l idy = threadIdx.y+blockIdx.y*blockDim.y;
    real_l idz = threadIdx.z+blockIdx.z*blockDim.z;

    real_l lx_min = const_args[0]+const_args[6]*idx;
    real_l ly_min = const_args[2]+const_args[7]*idy;
    real_l lz_min = const_args[4]+const_args[8]*idz;

    real_l temp_id = 0;
    cell_list[idx][idy][idz] = 0;

    //Iterate through all the particles and fill in both lists
    for(real_l i=0;i<num_particles;i++){
        if(InRange(lx_min,lx_min+const_args[6],position[3*i],const_args,0)\
           && InRange(ly_min,ly_min+const_args[7],position[3*i+1],const_args,1)\
           && InRange(lz_min,lz_min+const_args[8],position[3*i+2],const_args,2)){
            if(cell_list[idx][idy][idz] == 0){
                cell_list[idx][idy][idz] = i+1;
            }
            else{
                temp_id = cell_list[idx][idy][idz]-1;
                while(particle_list[temp_id] != 0){
                    temp_id = particle_list[temp_id]-1;
                };{
                    if(value >= x1 && value < x2){
                        return true;
                    }
                    else if(value ==  x2 && x2 == const_args[dim*2+1]){
                        return true;
                    }
                    return false;

                }
                particle_list[temp_id] = i+1;
                particle_list[i] = 0;
            }
        }
    }
}

//To update the linked lists after every iteration using a particle parallel approach. This kernel should be launched using 1d thread blocks
__global__ void updateListsPp(real_l *cell_list, real_l *particle_list, const real_d *position, const real_d *const_args, const real_l num_particles){
    real_l idx = threadIdx.x+blockIdx.x*blockDim.x;

    real_l index  = idx*3;

    //Finding the cell indices
    real_l x_id = position[index]/const_args[6];
    real_l y_id = position[index+1]/const_args[7];
    real_l z_id = position[index+2]/const_args[8];


}

__device__ bool InRange(real_d x1,real_d x2, real_d value, const  real_d  *const_args, int dim){
    if(value >= x1 && value < x2){
        return true;
    }
    else if(value ==  x2 && x2 == const_args[dim*2+1]){
        return true;
    }
    return false;

}
