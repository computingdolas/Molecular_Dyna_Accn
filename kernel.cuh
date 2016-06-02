#include <cuda_runtime.h>
#include <vector>
#include "Type.h"


// Calculation of the Leonard Jones Potential
__device__ void lenardJonesPotential(const real_d *relativeVector,real_d * forceVector ,const real_d sigma , const real_d epislon) ;

// Calculation of the minimum forces
__device__ void minDist(real_d* relativeVector,const real_d *p,const real_d *n, const real_d *cellLength) ;

// Calculating the norm of the distance
__device__ real_d norm(const real_d *vector) ;


// Calculation of the forces for brute force calculation
__global__ void  calcForces(real_d *force,const real_d *position,const real_l numParticles ,const real_d sigma, const real_d epislon,const real_d rcutoff, const real_d * const_args){


    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numParticles){
    real_l vidxp = idx * 3 ;

    // Relative Vector
    real_d relativeVector[3] = {0.0,0.0,0.0} ;

    //Force Vector
    real_d forceVector[3] = {0.0000000000,0.00000000000,0.0000000000} ;

    //Domain lengths
    real_d dom_length[3] = {0.0,0.0,0.0};
    for(real_l i=0;i<3;i++){
        dom_length[i] = const_args[i*2+1]-const_args[i*2];
    }
    // Initialising the force again to zero
    for (real_l i =0 ; i < numParticles * 3;++ i ){
	force[i] = 0.0 ; 
    }


    // for loop to initialise the vector with appropriate values
    for (real_l i = 0 ; i < numParticles ; ++i){
        if(i != idx){
            // Find out the index of particle
            real_l vidxn = i * 3 ;

            // Find out the realtive vector
            // Relative vector updation has to be done on the minDist device function and it has to be removed from here
            relativeVector[0] = position[vidxp] - position[vidxn] ;
            relativeVector[1] = position[vidxp+1] - position[vidxn+1] ;
            relativeVector[2] = position[vidxp+2] - position[vidxn+2] ;

            minDist(relativeVector,&position[vidxp],&position[vidxn], dom_length);

            // Find the magnitude of relative vector and if it is less than cuttoff radius , then potential is evaluated otherwise , explicitly zero force is given
            // Magnitude of relative vector
            if(norm(relativeVector) < rcutoff){
                // Find the force between these tow particle
                lenardJonesPotential(relativeVector,forceVector ,sigma,epislon) ;
                force[vidxp]   +=  forceVector[0] ;
                force[vidxp+1] +=  forceVector[1] ;
                force[vidxp+2] +=  forceVector[2] ;
            }
            else{
                force[vidxp]   +=  0.0 ;
                force[vidxp+1] +=  0.0 ;
                force[vidxp+2] +=  0.0 ;
            }

        }
    }
}
}

// Finding out the minimum distance between two particle keeping in mind the periodic boundary condition

__device__ void minDist(real_d* relativeVector,const real_d *p,const real_d *n, const real_d * cellLength) {//Here the cellLength is actually the domain length

    // Dummy distance
    real_d distold = norm(relativeVector) ;

    // Temporary relative vector array
    real_d tempRelativeVector[3] = {0.0} ;

    // Iterating over all 27 possibilities to find out the minimum distance
     for (real_d x = n[0] - cellLength[0] ; x <= n[0] + cellLength[0] ; x = x + cellLength[0] ){
         for (real_d y = n[1] - cellLength[1] ; y <= n[1] + cellLength[1] ; y = y+ cellLength[1]){
             for (real_d z = n[2] - cellLength[2] ; z <= n[2] + cellLength[2] ; z = z + cellLength[2]) {

                 // Finding out the relative vector
                 tempRelativeVector[0] = p[0] -x;
                 tempRelativeVector[1] = p[1] -y;
                 tempRelativeVector[2] = p[2] -z;

                 // Find out the modulus of the distance
                 real_d dist = norm(tempRelativeVector) ;

                 // Holding the minimum value and finding out the minimum of all of them

                 if(dist < distold){
                     relativeVector[0] = p[0] - x ;
                     relativeVector[1] = p[1] - y ;
                     relativeVector[2] = p[2] - z ;
                     distold = dist ;
                 }

            }
        }
     }

}

// Intialising the vector for cell length for x , y ,z directions
//real_d cellLength[3] = {0.0} ;
// Position update and taking care of periodic boundary conditions
__global__ void updatePosition(const real_d *force,real_d *position,const real_d* velocity, const real_d * mass,const real_l numparticles,const real_d timestep, const real_d* const_args) {

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles ){

        real_l vidx = idx * 3 ;

        position[vidx]   += (timestep * velocity[vidx] ) + ( (force[vidx] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+1] += (timestep * velocity[vidx+1] ) + ( (force[vidx+1] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+2] += (timestep * velocity[vidx+2] ) + ( (force[vidx+2] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;

        // CHecking if the particle has left the physical domain direction wise ,for each direction

        // Checking for the x direction
        if (position[vidx] < const_args[0]) position[vidx] += (const_args[1]-const_args[0]) ;
        if (position[vidx] > const_args[1]) position[vidx] -= (const_args[1]-const_args[0]) ;

        // Checking for the y direction
        if (position[vidx+1] < const_args[2]) position[vidx+1] += (const_args[3]-const_args[2]) ;
        if (position[vidx+1] > const_args[3]) position[vidx+1] -= (const_args[3]-const_args[2]) ;

        // CHecking for z direction and updating the same
        if (position[vidx+2] < const_args[4]) position[vidx+2] += (const_args[5]-const_args[4]) ;
        if (position[vidx+2] > const_args[5]) position[vidx+2] -= (const_args[5]-const_args[4]) ;

    }
}

// Velocity Update
__global__ void updateVelocity(const real_d*forceNew,const real_d*forceOld,real_d * velocity, const real_d* mass,const real_l numparticles ,const real_d timestep ){

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles){
    real_l vidx = idx * 3 ;

    velocity[vidx] += ( (forceNew[vidx] + forceOld[vidx]) * timestep ) / (2.0 * mass[idx] ) ;
    velocity[vidx+1] += ( (forceNew[vidx+1] + forceOld[vidx+1]) * timestep ) / (2.0 * mass[idx] ) ;
    velocity[vidx+2] += ( (forceNew[vidx+2] + forceOld[vidx+2]) * timestep ) / (2.0 * mass[idx] ) ;
    }

}

// Calculation of Leonard Jones Potential
__device__ void lenardJonesPotential(const real_d *relativeVector,real_d * forceVector ,const real_d sigma , const real_d epislon) {

    real_d distmod =  sqrt( (relativeVector[0] * relativeVector[0]) + (relativeVector[1] * relativeVector[1]) + (relativeVector[2] * relativeVector[2]) ) ;
    real_d dist = distmod * distmod ;
    real_d sigmaConstant =  sigma / distmod ;
    real_d epislonConstant =  (24.0 * epislon) / dist ;

    real_d con = ( (2.0 *(pow(sigmaConstant,6.0))) - 1.00000000 ) ;
    
    forceVector[0] = epislonConstant * pow(sigmaConstant,6.0) * con * relativeVector[0] ;
    forceVector[1] = epislonConstant * pow(sigmaConstant,6.0) * con * relativeVector[1] ;
    forceVector[2] = epislonConstant * pow(sigmaConstant,6.0) * con * relativeVector[2] ;
    	   
}

// Calculate the magnitude of the relative vector
__device__ real_d norm(const real_d *vector) {

    real_d sum = 0.0 ;
    for (real_l i =0 ; i < 3 ; ++i){
        sum  += vector[i] * vector[i] ;
    }
    return sqrt(sum) ;
}

// Copying the forces
__global__ void copyForces(real_d * fold,real_d * fnew, const real_l numparticles) {

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles){
    real_l vidxp = idx * 3 ;

    for(real_l i =vidxp ; i < vidxp+3; ++i ){
            fold[i] = fnew[i] ;
    }
    }
}



//To check if the given value lies betwen x1 and x2
__device__ bool InRange(real_d x1,real_d x2, real_d value, const  real_d  *const_args, int dim){
    if(value >= x1 && value < x2){
        return true;
    }
    else if(value ==  x2 && x2 == const_args[dim*2+1]){
        return true;
    }
    return false;

}


//Reset the buffer with 0s
__global__ void SetToZero(real_l *buffer, real_l buffer_size){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx < buffer_size){
        buffer[idx] = 0;
    }
}

//To update the linked lists after every iteration . This kernel should be launched using 3d thread blocks
__global__ void updateLists(real_l *cell_list, real_l *particle_list, const real_d *position, const real_d *const_args, const real_l num_particles){

    real_l idx = threadIdx.x+blockIdx.x*blockDim.x;
    real_l idy = threadIdx.y+blockIdx.y*blockDim.y;
    real_l idz = threadIdx.z+blockIdx.z*blockDim.z;
    
    real_l id_g = idz*(gridDim.x*blockDim.x*gridDim.y*blockDim.y) + idy*(gridDim.x*blockDim.x) + idx;

    //Finding the coordinates
    real_l lx_min = const_args[0]+const_args[6]*idx;
    real_l ly_min = const_args[2]+const_args[7]*idy;
    real_l lz_min = const_args[4]+const_args[8]*idz;

    real_l temp_id = 0;
    cell_list[id_g] = 0;

    real_l i = 0;
    //Iterate through all the particles and fill in both lists
    for(i=0;i<num_particles;i++){
        if(InRange(lx_min,lx_min+const_args[6],position[3*i],const_args,0)\
           && InRange(ly_min,ly_min+const_args[7],position[3*i+1],const_args,1)\
           && InRange(lz_min,lz_min+const_args[8],position[3*i+2],const_args,2)){
            if(cell_list[id_g] == 0){
                cell_list[id_g] = i+1;
            }
            else{
                temp_id = cell_list[id_g]-1;
                while(particle_list[temp_id] != 0){
                    temp_id = particle_list[temp_id]-1;
                }
                particle_list[temp_id] = i+1;
                particle_list[i] = 0;
            }
        }
    }
}


// Update the list in particle parallel
__global__ void updateListParticleParallel(real_l * cell_list, real_l * particle_list, const real_d  * const_args, const real_l num_particles ,const  real_d * position, const real_l * numcell  ) {

    real_l idx = threadIdx.x+blockIdx.x*blockDim.x;

    if (idx < num_particles) {

        // Finding the index of the particles
        real_l  vidxp = idx * 3 ;

        // Finding the cordinates of the particle
        real_d pos[3] = {position[vidxp], position[vidxp+1], position[vidxp+2] } ;

        // Find the ... cordinate of the cell it lies and register it their using atomic operations
        real_l i = pos[0] / const_args[6] ;
        real_l j = pos[1] / const_args[7] ;
        real_l k = pos[2]/ const_args[8];

        // Find the global id of the cell
        real_l cellindex = i + j * numcell[0] + numcell[0] * numcell[1] * k ;

        // See whether that cell has already has some master particle , and if not assign itself to it and
        real_l old = atomicExch(&cell_list[cellindex] ,idx+1);
        particle_list[idx] = old ;
    }
}

__global__ void formNeighbourList(real_l * neighbourList,const real_d * position, const real_d * const_args, const real_l numparticles,const real_d rcutoff ){

    real_l idx = threadIdx.x+blockIdx.x*blockDim.x;

    if(idx < numparticles ){

        // Relative Vector
        real_d relativeVector[3] = {0.0,0.0,0.0} ;

        // Domain length vector
        real_d dom_length[3] = {0.0,0.0,0.0} ;
        for(real_l i=0;i<3;i++){
            dom_length[i] = const_args[i*2+1]-const_args[i*2];
        }

        // Index of the particle forming the neighbour list
        real_l vidxp = idx * 3 ;

        // Index of particle in the neighbour list array
        real_l nidx = idx * numparticles ;
        real_l iter = 0 ;

        // Sweep across all particles
        for(real_l i =0 ; i < numparticles ; ++i){

            if( i !=idx ) {

                //Find out th index of the self and the neighbouring particles
                real_l vidxn = i * 3 ;

                // Find out the position of the neighbour particle
                relativeVector[0] = position[vidxp] - position[vidxn] ;
                relativeVector[1] = position[vidxp+1] - position[vidxn+1] ;
                relativeVector[2] = position[vidxp+2] - position[vidxn+2] ;

                // Find out the optimal distance , keeping in mind the periodic boundary condition in mind
                minDist(relativeVector,&position[vidxp],&position[vidxn],dom_length);

                // If norm of the relative vector is less than rcuttoff , then  just include the particle in the neighbour list - constant can we decided arbitarily
                if (norm(relativeVector) < (1.5 * rcutoff)){
                     neighbourList[nidx + iter] = i ;
                     ++iter ;
                }
            }
         }
        neighbourList[nidx + iter] = idx ;
     }
}

__global__ void calcForcesNeighbourList(real_d *force, const real_l * neighbourList,const real_l numParticles, const real_d * const_args,const real_d * position,const real_d sigma, const real_d epislon){

    // Calculate the index of the particle
    real_l idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < numParticles){

        // Relative Vector
        real_d relativeVector[3] = {0.0,0.0,0.0} ;

        // Domain length
        real_d dom_length[3] = {0.0,0.0,0.0} ;
        for(real_l i=0;i<3;i++){
            dom_length[i] = const_args[i*2+1]-const_args[i*2] ;
        }

        // Force vector dummy
        real_d forceVector[3] = {0.0,0.0,0.0} ;

        // Initialising the force again to zero , so that it doest not accumulate , into force array
        for (real_l i =0 ; i < numParticles * 3;++ i ){
        force[i] = 0.0 ;
        }

        //Index of particle calculating the forces due to other particles
        real_l vidxp = idx * 3 ;

        // Traverse the neighbour list uptill this particle encounter itself
        {

            // Declaring the iterator
            real_l i = 0 ;
            while (neighbourList[i] != idx){

                // Find out the index of particle in the particle array
                real_l vidxn = neighbourList[i] * 3 ;

                // Find out the relative vector between the two particle
                relativeVector[0] = position[vidxp] - position[vidxn]  ;
                relativeVector[1] = position[vidxp+1] - position[vidxn+1]  ;
                relativeVector[2] = position[vidxp+2] - position[vidxn+2]  ;

                // Find out the optimal distance keeping in mind the periodic boundary conditions
                minDist(relativeVector,&position[vidxp],&position[vidxn],dom_length) ;

                //  Find out the forces between two particles , we do not consider other case of non zero force
                lenardJonesPotential(relativeVector,forceVector ,sigma,epislon)  ;

                // Add the forces cumulatively
                force[vidxp]     +=  forceVector[0]  ;
                force[vidxp+1] +=  forceVector[1]  ;
                force[vidxp+2] +=  forceVector[2]  ;

                // Increase the iterator ..
                ++i ;
            }
        }
    }
}










