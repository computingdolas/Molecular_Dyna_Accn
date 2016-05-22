#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>
#include "Parser.h"
#include "PhysicalVariable.h"
#include "Type.h"
#include "kernel.cuh"
#include <string>
#include "VTKWriter.h"
#include <iomanip>
#include "Time.hpp"

int main(int argc, const char * argv[]) {
    // Reading from file
    Parser p(10,argv[1]);
    p.readParameters();
    p.readInputConfiguration();

    // number of Particles
    const real_l numparticles = p.num_particles ;

    // Creating the device Buffers
    cudaDeviceBuffer<real_d> mass(numparticles,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> position(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> velocity(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forceold(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forcenew(numparticles,PhysicalQuantity::Vector) ;

    p.fillBuffers(mass,velocity,position);


    // Allocating memory on Device
    mass.allocateOnDevice();
    position.allocateOnDevice();
    velocity.allocateOnDevice();
    forceold.allocateOnDevice();
    forcenew.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    forceold.copyToDevice();
    forcenew.copyToDevice();

    // Parameters from the file
    real_d time_end = std::stod(p.params["time_end"]) ;
    real_d timestep_length = std::stod(p.params["timestep_length"]) ;
    real_d epsilon = std::stod(p.params["epsilon"]) ;
    real_d sigma = std::stod(p.params["sigma"]) ;
    real_l vtk_out_freq = std::stol(p.params["vtk_out_freq"]) ;
    real_l threads_per_blocks = std::stol(p.params["cl_workgroup_1dsize"]) ;
    std::string vtk_name = p.params["vtk_out_name_base"] ;

    VTKWriter writer(vtk_name) ;

    //Calculate the number of blocks
    real_l num_blocks ;

    if(numparticles % threads_per_blocks ==0) num_blocks = numparticles / threads_per_blocks ;
    else num_blocks = (numparticles / threads_per_blocks) + 1 ;

    //std::cout<<num_blocks<<" "<<threads_per_blocks<<std::endl;
    real_d time_taken = 0.0 ;

    HESPA::Timer time ;
    // Algorithm to follow
    {

        real_l iter = 0 ;
        // calculate Initial forces
        calcForces<<<num_blocks ,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,numparticles,sigma,epsilon) ;
        for(real_d t =0.0 ; t < time_end ; t+= timestep_length ) {
            time.reset();
            // Update the Position
            updatePosition<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,velocity.devicePtr,mass.devicePtr,numparticles,timestep_length);

            // Copy the forces
            copyForces<<<num_blocks,threads_per_blocks>>>(forceold.devicePtr,forcenew.devicePtr, numparticles);

            // Calculate New forces
            calcForces<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,numparticles, sigma,epsilon);

            // Update the velocity
            updateVelocity<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,forceold.devicePtr,velocity.devicePtr,mass.devicePtr,numparticles,timestep_length);

            cudaDeviceSynchronize();
            time_taken += time.elapsed();

            if(iter % vtk_out_freq == 0){
                // copy to host back
                forcenew.copyToHost();
                forceold.copyToHost();
                position.copyToHost();
                velocity.copyToHost();
                writer.writeVTKOutput(mass,position,velocity,numparticles);
            }

            // Iterator count
            ++iter ;
        }

    }

    std::cout<<"The time taken for "<<numparticles<<" is:= "<<time_taken<<std::endl ;

    return 0;
}                           
