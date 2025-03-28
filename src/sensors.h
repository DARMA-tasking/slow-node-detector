#ifndef SRC_SENSORS_H
#define SRC_SENSORS_H

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
  std::vector<double>& all_max_temps,
  std::vector<int>& all_socket_orders,
  std::vector<int>& all_core_orders,
  std::vector<int>& all_num_values,
  std::vector<int>& all_node_ids,
  std::vector<int>& all_cpu_freqs,
  std::map<int, std::string>& node_map,
  std::string identifier
);

// Main function
void runSensorsAndReduceOutput(
  const std::string& proc_name,
  std::string identifier = "");

} // namespace sensors

#endif // SRC_SENSORS_H
