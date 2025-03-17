#ifndef SRC_SENSORS_HPP
#define SRC_SENSORS_HPP

#include <map>
#include <vector>
#include <string>

#include <mpi.h>

namespace sensors {

int getNodeID(MPI_Comm initial_communicator);

std::map<int, std::map<int,double>> parseSensorsOutput(FILE* pipe);

std::map<int, std::map<int, double>> runSensors();

void getTempsAndOrdering(
  std::map<int, std::map<int, double>> packageCoreTemps,
  std::vector<double>& local_temps,
  std::vector<std::pair<int,int>>& ordering
);

void writeSensorData(
  std::vector<double>& max_temps,
  std::vector<std::pair<int,int>>& ordering,
  const std::string& proc_name
);

// Main function
void runSensorsAndReduceOutput(const std::string& proc_name);

} // namespace sensors

#endif // SRC_SENSORS_HPP
