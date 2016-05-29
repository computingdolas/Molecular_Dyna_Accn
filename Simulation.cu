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
    Parser p(argv[1]);
    p.readParameters();
    p.readInputConfiguration();

    // Parameters from the file
    real_d time_end = std::stod(p.params["time_end"]) ;
    real_d timestep_length = std::stod(p.params["timestep_length"]) ;
    real_d epsilon = std::stod(p.params["epsilon"]) ;
    real_d sigma = std::stod(p.params["sigma"]) ;
    real_d r_cut = std::stod(p.params["r_cut"]);
    real_l vtk_out_freq = std::stol(p.params["vtk_out_freq"]) ;
    real_l threads_per_blocks = std::stol(p.params["cl_workgroup_1dsize"]) ;
    std::string vtk_name = p.params["vtk_out_name_base"] ;
    real_d xmin = std::stod(p.params["xmin"]);
    real_d xmax = std::stod(p.params["xmax"]);
    real_d ymin = std::stod(p.params["ymin"]);
    real_d ymax = std::stod(p.params["ymax"]);
    real_d zmin = std::stod(p.params["zmin"]);
    real_d zmax = std::stod(p.params["zmax"]);
    real_l xn = std::stol(p.params["xn"]);
    real_l yn = std::stol(p.params["yn"]);
    real_l zn = std::stol(p.params["zn"]);

    real_l len_x = (x_max-x_min)/xn;
    real_l len_y = (y_max-y_min)/yn;
    real_l len_z = (z_max-z_min)/zn;

    // number of Particles
    const real_l numparticles = p.num_particles ;
    const real_l numcells = xn*yn*zn;

    // Creating the device Buffers
    cudaDeviceBuffer<real_d> mass(numparticles,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> position(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> velocity(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forceold(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forcenew(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_l> cell_list(numcells,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<real_l> particle_list(numparticles,PhysicalQuantity::Scalar);

    //Initiliazing the buffers for mass,velocity and position
    p.fillBuffers(mass,velocity,position);


    // Allocating memory on Device
    mass.allocateOnDevice();
    position.allocateOnDevice();
    velocity.allocateOnDevice();
    forceold.allocateOnDevice();
    forcenew.allocateOnDevice();
    cell_list.allocateOnDevice();
    particle_list.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    forceold.copyToDevice();
    forcenew.copyToDevice();
    cell_list.copyToDevice();
    particle_list.copyToDevice();


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
        calcForces<<<num_blocks ,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,numparticles,sigma,epsilon,r_cut) ;
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
