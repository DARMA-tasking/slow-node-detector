#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <cassert>
#include <numeric>
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

void getTempsAndOrders(
  std::map<int, std::map<int, double>> socketCoreTemps,
  std::vector<double>& local_temps,
  std::vector<int>& socket_order,
  std::vector<int>& core_order
) {
  for (const auto& [socketId, coreMap] : socketCoreTemps) {
    for (const auto& [coreNumber, temperature] : coreMap) {
      local_temps.push_back(temperature);
      socket_order.push_back(socketId);
      core_order.push_back(coreNumber);
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
  std::vector<int> socket_order; // this is the same for all ranks on this node
  std::vector<int> core_order;   // this is the same for all ranks on this node
  getTempsAndOrders(socketCoreTemps, local_temps, socket_order, core_order);

  // Reduce with MPI_MAX across the node communicator
  int num_values = local_temps.size();
  std::vector<double> max_temps(num_values, -1e9);
  MPI_Reduce(local_temps.data(), max_temps.data(), num_values, MPI_DOUBLE, MPI_MAX, 0, node_comm);

  // Format results on rank 0 of current node
  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  int global_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  std::cout << "Global: " << global_rank << ",  Node: " << node_rank << std::endl;

  // node_rank 0 is holding three vectors:
  //    max_temps - a reduced vector that holds all the max temps for each core on this node
  //    socket_order - a vector of socket IDs that corresponds one-to-one with each temperature
  //    core_order - a vector of core IDs taht corresponds one-to-one with each temperature
  //
  // for example, max_temps[i] is on core core_order[i] of socket socket_order[i]
  //
  // node_rank 0 is also holding num_values, an int that says how many cores/sockets/temps are in those vectors

  // global rank 0 needs to concatenate all of these vectors from the node_rank 0s
  //    -> This is using gatherv, where the size of the buffer is given with num_values

  // first, we gather all of the num_values into a vector

  int n_nodes = 1; // hard-code for now
  std::vector<int> all_num_values(n_nodes);
  if (node_rank == 0) {
    MPI_Gather(&num_values, 1, MPI_INT,
               all_num_values.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int total_num_entries = std::accumulate(all_num_values.begin(), all_num_values.end(), 0);
  std::vector<int> displacements(n_nodes, 0);
  std::vector<double> all_max_temps(total_num_entries);
  if (node_rank == 0) {
    for (int i=1; i<n_nodes; i++) {
      displacements[i] = displacements[i-1] + all_num_values[i-1];
    }
    std::cout << " Testing from rank " << global_rank << std::endl;
    for (const auto& elt : displacements) {
      std::cout << elt << std::endl;
    }
    MPI_Gatherv(max_temps.data(), num_values, MPI_DOUBLE,
                all_max_temps.data(), all_num_values.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
  }

  // so global rank 0 would end up with three vectors:
  //    all_max_temps - a vector containing all of the values of max_temps vectors, IN ORDER
  //    all_socket_order - a vector containing all of the IDs from socket_order vectors, IN ORDER
  //    all_core_order - a vector containing all of the IDs from core_order vectors, IN ORDER
  //
  //    num_values_per_node - a vector containing the number of values that came from each node
  //        so if there are two nodes with 16 cores each, this would be [ 16, 16]

  // global rank 0 can then iterate through the all_max_temps vector, matching with the socket and core from the order vectors
  // once num_values_per_node[i] is reached, a newline is created and a new node begins

  // the only remaining problem is getting the node name on rank 0

  // Write the results on rank 0 of the current node
  // int node_rank;
  // MPI_Comm_rank(node_comm, &node_rank);
  // if (node_rank == 0) {
  //   writeSensorData(max_temps, socket_order, core_order, proc_name);
  // }
  // MPI_Comm_free(&node_comm);
}

} // namespace sensors
