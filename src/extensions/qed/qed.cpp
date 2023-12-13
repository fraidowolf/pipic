/*-------------------------------------------------------------------------------------------------------
This file is part of pi-PIC.
pi-PIC, Copyright 2023 Joel Magnusson
---------------------------------------------------------------------------------------------------------
pi-PIC is free software: you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

pi-PIC is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with pi-PIC. If not, se
<https://www.gnu.org/licenses/>.
---------------------------------------------------------------------------------------------------------
Website: https://github.com/hi-chi/pipic
Contact: arkady.gonoskov@gu.se.
-------------------------------------------------------------------------------------------------------*/
// Description: A C/C++ extension that can be used to account for quantum radiation reaction.

#include "interfaces.h"
#include <pybind11/pybind11.h>
#include "pybind11/stl.h"
#include <pybind11/operators.h>

const string name = "qed";
static int photonTypeId, electronTypeId, positronTypeId; // particle types

void onePhotonStep(particle &P, const double3 &E, const double3 &B, double &time, double timeStep, int type, vector<particle> &AvalancheParticles, vector<double> &timeAvalancheParticles, vector<int> &typeAvalancheParticles) {
    double SchwingerField = sqr(constants::electronMass * constants::lightVelocity) * constants::lightVelocity
                            / (-constants::electronCharge * constants::hbar);
    double c = constants::lightVelocity;
    double mc = constants::electronMass * constants::lightVelocity; // Change to real particle mass

    while (time < timeStep) {
        double p = P.p.norm();
        double gamma = p/mc; // gamma = E/mc2 = |p|/mc

        double3 k_ = P.p/p; // normalized wave vector

        double H_eff = sqrt(sqr(E + cross(k_, B)) - sqr(dot(E, k_)));
        double chi = gamma * H_eff / SchwingerField;
        double rate = 0.0, dt = 2*timeStep;

        // Compute rate and subtimestep (dt)
        if (chi > 0.0) {
            // rate = breit_wheeler.rate(chi);
            // dt = getDtPhoton(particle, rate, chi, gamma);
        }

        if (dt + time > timeStep) {
            // Move particle to end of timeStep. No particle push being carried out in this part of code
            time = timeStep;
        } else {
            // Move particle by a subtimestep (dt). No particle push being carried out in this part of code
            time += dt;

            // determine new particle energy
            double delta = 0.5; // Pair_Generator(chi);

            // Create new particle (electron)
            particle new_particle;
            new_particle.w = P.w;
            new_particle.r = P.r;
            new_particle.p = delta * P.p;

            // Add new particle to container for later processing
            AvalancheParticles.push_back(new_particle);
            timeAvalancheParticles.push_back(time);
            typeAvalancheParticles.push_back(electronTypeId);

            // Create new particle (positron)
            new_particle.p = (1-delta) * P.p;

            // Add new particle to container for later processing
            AvalancheParticles.push_back(new_particle);
            timeAvalancheParticles.push_back(time);
            typeAvalancheParticles.push_back(positronTypeId);

            // Change current particle momentum
            P.p = 0.0 * P.p;
            P.w = 0.0; // Marks particle for deletion
        }
    }
}

void oneParticleStep(particle &P, const double3 &E, const double3 &B, double &time, double timeStep, int type, vector<particle> &AvalancheParticles, vector<double> &timeAvalancheParticles, vector<int> &typeAvalancheParticles) {
    double SchwingerField = sqr(constants::electronMass * constants::lightVelocity) * constants::lightVelocity
                            / (-constants::electronCharge * constants::hbar);
    double c = constants::lightVelocity;
    double mc = constants::electronMass * constants::lightVelocity; // Change to real particle mass

    while (time < timeStep) {
        double gamma = sqrt(1 + sqr(P.p)/sqr(mc));
        double3 beta = P.p / (gamma*mc);

        double H_eff = sqr(E + cross(beta, B)) - sqr(dot(E, beta));
        if (H_eff < 0) H_eff = 0;
        H_eff = sqrt(H_eff);

        double chi = gamma * H_eff / SchwingerField;
        double rate = 0.0, dt = 2*timeStep;

        // Compute rate and subtimestep (dt)
        if (chi > 0.0) {
            // rate = compton.rate(chi);
            // dt = getDtParticle(particle, rate, chi);
        }

        if (dt + time > timeStep) {
            // Move particle to end of timeStep. No particle push being carried out in this part of code
            time = timeStep;
        } else {
            // Move particle by a subtimestep (dt). No particle push being carried out in this part of code
            time += dt;

            // determine new particle energy
            double delta = 0.5; // Photon_Generator((chi + chi_new)/(FP)2.0);

            // Create new particle (photon)
            particle new_particle;
            new_particle.w = P.w;
            new_particle.r = P.r;
            new_particle.p = delta * P.p;

            // Add new particle to container for later processing
            AvalancheParticles.push_back(new_particle);
            timeAvalancheParticles.push_back(time);
            typeAvalancheParticles.push_back(photonTypeId);

            // Change current particle momentum
            P.p = (1 - delta) * P.p;
        }
    }
}


void RunAvalanche(const double3 E, const double3 B, double timeStep, vector<particle> &AvalancheParticles, vector<double> &timeAvalancheParticles, vector<int> &typeAvalancheParticles) {
    // Track the number of processed particles
    int countParticles = 0;

    while (countParticles != AvalancheParticles.size() || false) {
        for (int k = countParticles; k!=AvalancheParticles.size(); k++) {
            int typeId = typeAvalancheParticles[k];
            // Process a single particle a full timeStep
            if (typeId == electronTypeId || typeId == positronTypeId) {
                oneParticleStep(AvalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep, typeId, AvalancheParticles, timeAvalancheParticles, typeAvalancheParticles);
                countParticles++;
            } else if (typeId == photonTypeId) {
                onePhotonStep(AvalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep, typeId, AvalancheParticles, timeAvalancheParticles, typeAvalancheParticles);
                countParticles++;
            }
        }
    }
}

// function that is called to process particles in a given cell
void Handler(int *I, double *D, double *F, double *P, double *NP, double *dataDouble, int *dataInt) {
    // interface for manipulating with the content of a cell
    cellInterface CI(I, D, F, P, NP);

    if (CI.particleSubsetSize > 0) {
        // Vectors for storing particles during processing
        vector<particle> AvalancheParticles;
        vector<double> timeAvalancheParticles;
        vector<int> typeAvalancheParticles;

        // Vector for storing particles after processing
        vector<particle> AfterAvalancheParticles;

        for(int ip = 0; ip < CI.particleSubsetSize; ip++) {
            // Grab a single particle from CellInterface
            particle *P = CI.Particle(ip);

            // Compute the field at particle position (E, B are passed by reference)
            double3 E, B;
            CI.interpolateField(P->r, E, B);

            // Clear particle containers
            AvalancheParticles.clear();
            timeAvalancheParticles.clear();
            typeAvalancheParticles.clear();

            // Place the particle in the container
            AvalancheParticles.push_back(*P);
            timeAvalancheParticles.push_back(0.0);
            typeAvalancheParticles.push_back(CI.particleTypeIndex);

            // Run a full timeStep for the particle (including for its daughter particles)
            RunAvalanche(E, B, CI.timeStep, AvalancheParticles, timeAvalancheParticles, typeAvalancheParticles);

            // Update particle momentum and weight (in case of removal)
            P->p = AvalancheParticles[0].p;
            P->w = AvalancheParticles[0].w;

            // Place all created particles in the buffer
            for (int k = 1; k < AvalancheParticles.size(); k++) {
                // Make sure that particle isn't marked for deletion
                if (AvalancheParticles[k].w > 0.0) {
                    // Check if the buffer permits adding a particle
                    if(CI.particleBufferSize < CI.particleBufferCapacity){
                        // Copy particle to the new-particle buffer
                        *CI.newParticle(CI.particleBufferSize) = AvalancheParticles[k];
                        // Set new-particle type id
                        CI.newParticle(CI.particleBufferSize)->id = typeAvalancheParticles[k];
                        CI.particleBufferSize++;
                    }
                }
            }
        }
    }
};

// extension initialization
int64_t handler(int photonType, int electronType, int positronType){
    photonTypeId = photonType;
    electronTypeId = electronType;
    positronTypeId = positronType;
    return (int64_t)Handler;
};

namespace py = pybind11;
PYBIND11_MODULE(_qed, object) {
    object.attr("name") = name;
    object.def("handler", &handler, py::arg("photon_type"), py::arg("electron_type"), py::arg("positron_type"));
}