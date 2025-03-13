#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cassert>
#include <functional>

#include "sensors.h"

namespace sensors {

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
    std::filesystem::path output_dir = "sensors_output";
    std::filesystem::path reduced_filename = output_dir / ("sensors_" + proc_name + ".log");
    std::filesystem::create_directories(output_dir);
    std::ofstream reduced_file(reduced_filename);
    if (!reduced_file) {
      std::cerr << "Error: Unable to open " << reduced_filename << " for writing.\n";
      return;
    }

    // Use the ordering vector to map back to socket and core IDs.
    for (size_t i = 0; i < ordering.size(); i++) {
      auto [socketId, coreNumber] = ordering[i];
      reduced_file << "Socket id " << socketId
                        << ", Core " << coreNumber
                        << ": " << max_temps[i] << " C\n";
    }
    reduced_file.close();
    std::cout << "Wrote sensor data for " << proc_name << " to " << reduced_filename << std::endl;
}

void runSensorsAndReduceOutput(int rank, const std::string& proc_name) {
  // Set up
  size_t proc_hash = std::hash<std::string>{}(proc_name);
  int color = static_cast<int>(proc_hash % std::numeric_limits<int>::max());
  MPI_Comm node_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &node_comm);
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
  MPI_Reduce(local_temps.data(), max_temps.data(), num_values, MPI_DOUBLE, MPI_MAX, 0, node_comm);

  // Write the results on rank 0 of the current node
  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);
  if (node_rank == 0) {
    writeSensorData(max_temps, ordering, proc_name);
  }
}

} // namespace sensors
