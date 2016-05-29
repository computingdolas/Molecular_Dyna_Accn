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

    // Initialising the force again to zero
    for (real_l i =0 ; i < numParticles * 3;++ i ){
	force[i] = 0.0 ; 
    }

    // Intialising the vector for cell length for x , y ,z directions
    real_d cellLength[3] = {0.0} ;

    // for loop to initialise the vector with appropriate values

    for (real_l i = 0 ; i < numParticles ; ++i){
        if(i != idx){
            // Find out the index of particle
            real_l vidxn = i * 3 ;

            // Find out the realtive vector
            // Relative vector updation has to be done on the minDist device function and it has to be removed from here
            minDist(relativeVector,&position[vidxp],&position[vidxn], &const_args[6]);

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

// Finding out the minimum distance between two particle keeping in mind the periodic boundary condition

__device__ void minDist(real_d* relativeVector,const real_d *p,const real_d *n, const real_d * cellLength) {

    // Dummy distance
    real_d distold = cellLength[0] * cellLength[0] * cellLength[0] ;

    // Temporary relative vector array
    real_d tempRelativeVector[3] = {0.0} ;

    // Iterating over all 27 possibilities to find out the minimum distance
     for (real_d x = n[0] - cellLength[0] ; x <= n[0] + cellLength[0] ; x = x + cellLength[0] )
         for (real_d y = n[1] - cellLength[1] ; y <= n[1] + cellLength[1] ; y = y+ celllength[1])
             for (real_d z = n[2] - cellLength[2] ; z <= n[2] + cellLength[2] ; z = z + cellLength[2]) {

                 // Finding out the relative vector
                 tempRelativeVector[0] = p[0] - x ;
                 tempRelativeVector[1] = p[1] - y ;
                 tempRelativeVector[2] = p[2] - z ;

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


// Position update and taking care of periodic boundary conditions
__global__ void updatePosition(const real_d *force,real_d *position,const real_d* velocity, const real_d * mass,const real_l numparticles,const real_d timestep, const
                                real_d * const_args) {

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;
    if(idx < numparticles ){

        real_l vidx = idx * 3 ;

        position[vidx]   += (timestep * velocity[vidx] ) + ( (force[vidx] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+1] += (timestep * velocity[vidx+1] ) + ( (force[vidx+1] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+2] += (timestep * velocity[vidx+2] ) + ( (force[vidx+2] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;

        // CHecking if the particle has left the physical domain direction wise ,for each direction

        // Checking for the x direction
        if (position[vidx] <= const_args[0]) position[vidx] += const_args[6] ;
        if (position[vidx] >= const_args[1]) position[vidx] -= const_args[6] ;

        // Checking for the y direction
        if (position[vidx+1] <= const_args[2]) position[vidx] += const_args[7] ;
        if (position[vidx+1] >= const_args[3]) position[vidx] -= const_args[7] ;

        // CHecking for z direction and updating the same
        if (position[vidx+2] <= const_args[4]) position[vidx] += const_args[8] ;
        if (position[vidx+2] >= const_args[5]) position[vidx] -= const_args[8] ;

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


