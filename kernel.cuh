#include <cuda_runtime.h>
#include <vector>
#include "Type.h"


//Calculation of the Leonard Jones Potential
__device__ void lenardJonesPotential(const real_d *relativeVector,real_d * forceVector ,const real_d sigma , const real_d epislon) ;

// Calculation of the forces
__global__ void  calcForces(real_d *force,const real_d *position,const real_l numParticles ,const real_d sigma, const real_d epislon){


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

    for (real_l i = 0 ; i < numParticles ; ++i){
        if(i != idx){
            // Find out the index of particle
            real_l vidxn = i * 3 ;

            // Find out the realtive vector
            relativeVector[0] = position[vidxp] - position[vidxn] ;
            relativeVector[1] = position[vidxp+1] - position[vidxn+1] ;
            relativeVector[2] = position[vidxp+2] - position[vidxn+2] ;

            // Find put the force between these tow particle
            lenardJonesPotential(relativeVector,forceVector ,sigma,epislon) ;
            force[vidxp] +=    forceVector[0] ;
            force[vidxp+1] +=  forceVector[1] ;
            force[vidxp+2] +=  forceVector[2] ;

        }
    }
    }
}

// Position update
__global__ void updatePosition(const real_d *force,real_d *position,const real_d* velocity, const real_d * mass,const real_l numparticles,const real_d timestep ) {

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;
    if(idx < numparticles ){
    real_l vidx = idx * 3 ;

    //(timestep * velocity[vidx] )   
    position[vidx]   += (timestep * velocity[vidx] ) + ( (force[vidx] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
    position[vidx+1] += (timestep * velocity[vidx+1] ) + ( (force[vidx+1] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
    position[vidx+2] += (timestep * velocity[vidx+2] ) + ( (force[vidx+2] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
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


