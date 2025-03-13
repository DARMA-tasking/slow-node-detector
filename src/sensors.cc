#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cassert>
#include <functional>

#include "sensors.h"

namespace sensors {

MPI_Comm getNodeCommunicator(MPI_Comm initial_communicator) {
  MPI_Comm shm_comm;
  MPI_Comm_split_type(initial_communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
  int shm_rank = -1;
  int node_size = -1;
  MPI_Comm_rank(shm_comm, &shm_rank);
  MPI_Comm_size(shm_comm, &node_size);

  int num_nodes = -1;
  int is_rank_0 = (shm_rank == 0) ? 1 : 0;
  MPI_Allreduce(&is_rank_0, &num_nodes, 1, MPI_INT, MPI_SUM, initial_communicator);

  int starting_rank = -1;
  MPI_Comm_rank(initial_communicator, &starting_rank);

  MPI_Comm node_communicator;
  MPI_Comm_split(initial_communicator, shm_rank, starting_rank, &node_communicator);

  MPI_Comm_free(&shm_comm);
  return node_communicator;
}

std::map<int, std::map<int,double>> parseSensorsOutput(FILE* pipe) {
  std::map<int, std::map<int,double>> socketCoreTemps;
  char buffer[256];
  int currentSocket = -1;

  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string line(buffer);

    if (line.find("Package id") != std::string::npos) {
      std::istringstream iss(line);
      std::string package_id;
      int socketId;
      char colon;
      iss >> package_id    // "Package"
          >> package_id    // "id"
          >> socketId      // socket number
          >> colon;        // the colon after the socket id
      currentSocket = socketId;
      if (socketCoreTemps.find(currentSocket) == socketCoreTemps.end()) {
        socketCoreTemps[currentSocket] = std::map<int,double>();
      }
    } else if (line.find("Core") != std::string::npos && currentSocket != -1) {
      std::istringstream iss(line);
      std::string coreWord;
      int coreNumber;
      char colon;
      std::string tempStr;

      if (!(iss >> coreWord >> coreNumber >> colon >> tempStr))
        continue;

      // Remove temperature unit from string
      size_t pos = tempStr.find("°C");
      if (pos != std::string::npos) {
        tempStr = tempStr.substr(0, pos);
      }
      pos = tempStr.find(" C");
      if (pos != std::string::npos) {
        tempStr = tempStr.substr(0, pos);
      }

      // Store the temperature as a double
      double temperature = std::stod(tempStr);
      auto& coreMap = socketCoreTemps[currentSocket];
      assert(coreMap.find(coreNumber) == coreMap.end()); // sanity check
      coreMap[coreNumber] = temperature;
    }
  }
  return socketCoreTemps;
}

std::map<int, std::map<int, double>> runSensors() {
  FILE* pipe = popen("sensors", "r");
  if (!pipe) {
    std::cerr << "Error: Unable to run sensors command\n";
    return {};
  }
  auto socketCoreTemps = parseSensorsOutput(pipe);
  pclose(pipe);
  return socketCoreTemps;
}

void getTempsAndOrdering(
  std::map<int, std::map<int, double>> socketCoreTemps,
  std::vector<double>& local_temps,
  std::vector<std::pair<int,int>>& ordering
) {
  for (const auto& [socketId, coreMap] : socketCoreTemps) {
    for (const auto& [coreNumber, temperature] : coreMap) {
      local_temps.push_back(temperature);
      ordering.emplace_back(socketId, coreNumber);
    }
  }
}

void writeSensorData(
  std::vector<double>& max_temps,
  std::vector<std::pair<int,int>>& ordering,
  const std::string& proc_name
) {
  std::filesystem::path reduced_filename = "sensors.log";
  std::ofstream reduced_file(reduced_filename);
  if (!reduced_file) {
    std::cerr << "Error: Unable to open " << reduced_filename << " for writing.\n";
    return;
  }

  // Use the ordering vector to map back to socket and core IDs.
  reduced_file << "Node: " << proc_name;
  for (size_t i = 0; i < ordering.size(); i++) {
    auto [socketId, coreNumber] = ordering[i];
    reduced_file << "Socket id " << socketId
                 << ", Core "    << coreNumber
                 << ": "         << max_temps[i] << " C\n";
  }
  reduced_file.close();
  std::cout << "Wrote sensor data for " << proc_name << " to " << reduced_filename << std::endl;
}

void runSensorsAndReduceOutput(const std::string& proc_name) {
  // Set up
  MPI_Comm node_comm = getNodeCommunicator(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // Get output from `sensors`
  auto socketCoreTemps = runSensors();
  if (socketCoreTemps.empty()) {
    return;
  }

  // Flatten the parsed data into a vector for MPI_Reduce
  std::vector<double> local_temps;
  std::vector<std::pair<int,int>> ordering;
  getTempsAndOrdering(socketCoreTemps, local_temps, ordering);

  // Reduce with MPI_MAX across the node communicator
  int num_values = local_temps.size();
  std::vector<double> max_temps(num_values, -1e9);
  // Make this an Allreduce (only write one file)
  MPI_Reduce(local_temps.data(), max_temps.data(), num_values, MPI_DOUBLE, MPI_MAX, 0, node_comm);

  // Write the results on rank 0 of the current node
  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);
  if (node_rank == 0) {
    writeSensorData(max_temps, ordering, proc_name);
  }
  MPI_Comm_free(&node_comm);
}

} // namespace sensors
