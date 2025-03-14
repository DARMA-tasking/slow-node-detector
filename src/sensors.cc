#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <functional>

#include "sensors.h"

namespace sensors {

void getNodeInformation(
  MPI_Comm i_comm,
  int& physical_node_id,
  int& physical_num_nodes,
  int& physical_node_size,
  int& physical_node_rank
) {
  MPI_Comm shm_comm;
  MPI_Comm_split_type(i_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
  int shm_rank = -1;
  int node_size = -1;
  MPI_Comm_rank(shm_comm, &shm_rank);
  MPI_Comm_size(shm_comm, &node_size);

  int num_nodes = -1;
  int is_rank_0 = (shm_rank == 0) ? 1 : 0;
  MPI_Allreduce(&is_rank_0, &num_nodes, 1, MPI_INT, MPI_SUM, i_comm);

  int starting_rank = -1;
  MPI_Comm_rank(i_comm, &starting_rank);

  MPI_Comm node_number_comm;
  MPI_Comm_split(i_comm, shm_rank, starting_rank, &node_number_comm);

  int node_id = -1;
  if (shm_rank == 0) {
    MPI_Comm_rank(node_number_comm, &node_id);
  }
  MPI_Bcast(&node_id, 1, MPI_INT, 0, shm_comm);

  MPI_Comm_free(&shm_comm);
  MPI_Comm_free(&node_number_comm);

  physical_node_id = node_id;
  physical_num_nodes = num_nodes;
  physical_node_size = node_size;
  physical_node_rank = shm_rank;
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
  std::vector<double>& all_max_temps,
  std::vector<int>& all_socket_orders,
  std::vector<int>& all_core_orders,
  int num_values
) {
  std::filesystem::path reduced_filename = "sensors.log";
  std::ofstream reduced_file(reduced_filename);
  if (!reduced_file) {
    std::cerr << "Error: Unable to open " << reduced_filename << " for writing.\n";
    return;
  }

  // Use the ordering vectors to map back to socket and core IDs.
  int iter = 0;
  for (size_t i = 0; i < all_max_temps.size(); i++) {
    if (i % num_values == 0) {
      reduced_file << "\nNode: " << iter << "\n";
      iter++;
    }
    reduced_file << "Socket id " << all_socket_orders[i]
                 << ", Core "    << all_core_orders[i]
                 << ": "         << all_max_temps[i] << " C\n";
  }
  reduced_file.close();
  std::cout << "Wrote sensor data to " << reduced_filename << std::endl;
}

void runSensorsAndReduceOutput(const std::string& proc_name) {
  // Determine the global rank
  int global_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

  // Get node info and comm
  int node_id, num_nodes, node_size, node_rank;
  getNodeInformation(MPI_COMM_WORLD, node_id, num_nodes, node_size, node_rank);
  MPI_Comm node_comm;
  MPI_Comm_split(MPI_COMM_WORLD, node_id, global_rank, &node_comm);

  // Determine which ranks will be "leaders"
  int global_leader = 0;
  int node_leader = 1;

  // Get output from `sensors`
  auto socketCoreTemps = runSensors();
  if (socketCoreTemps.empty()) {
    return;
  }

  // Flatten the parsed data into a vector for MPI_Reduce
  std::vector<double> local_temps;  // this may change from core to core on this node
  std::vector<int> socket_order;    // this is the same for all ranks on this node
  std::vector<int> core_order;      // this is the same for all ranks on this node
  getTempsAndOrders(socketCoreTemps, local_temps, socket_order, core_order);

  // Reduce with MPI_MAX across the node communicator
  int num_values = local_temps.size(); // this will be the same for every node
  std::vector<double> max_temps(num_values, -1e9);
  MPI_Reduce(local_temps.data(), max_temps.data(), num_values, MPI_DOUBLE, MPI_MAX, node_leader, node_comm);

  /*
   * On each node, node_leader is now holding three vectors:
   *    max_temps - a reduced vector that holds all the max temps for each core on this node
   *    socket_order - a vector of socket IDs that corresponds one-to-one with each temperature
   *    core_order - a vector of core IDs that corresponds one-to-one with each temperature
   *
   * for example, max_temps[i] is for core <core_order[i]> of socket <socket_order[i]>
   *
   * node_leader is also holding num_values, an int that says how many cores/sockets/temps are in those vectors

   * global_leader needs to concatenate all of these vectors from the node_leaders
   */

   int total_num_entries = num_values * num_nodes;
  std::vector<int> all_num_values(num_nodes, num_values);
  std::vector<int> displacements(num_nodes);
  std::vector<double> all_max_temps(total_num_entries);
  std::vector<int> all_socket_orders(total_num_entries);
  std::vector<int> all_core_orders(total_num_entries);
  MPI_Barrier(MPI_COMM_WORLD);
  if (node_rank == node_leader) {
    for (int i=1; i<num_nodes; i++) {
      displacements[i] = displacements[i-1] + all_num_values[i-1];
    }
    MPI_Gatherv(max_temps.data(), num_values, MPI_DOUBLE,
                all_max_temps.data(), all_num_values.data(), displacements.data(), MPI_DOUBLE,
                global_leader, MPI_COMM_WORLD);
    MPI_Gatherv(socket_order.data(), num_values, MPI_INT,
                all_socket_orders.data(), all_num_values.data(), displacements.data(), MPI_INT,
                global_leader, MPI_COMM_WORLD);
    MPI_Gatherv(core_order.data(), num_values, MPI_INT,
                all_core_orders.data(), all_num_values.data(), displacements.data(), MPI_INT,
                global_leader, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /*
   * so global_leader now holds three vectors:
   *    all_max_temps - a vector containing all of the values of max_temps vectors, IN ORDER
   *    all_socket_orders - a vector containing all of the IDs from socket_order vectors, IN ORDER
   *    all_core_orders - a vector containing all of the IDs from core_order vectors, IN ORDER
   *
   *    all_num_values - a vector containing the number of values that came from each node
   *        so if there are two nodes with 16 cores each, this would be [16, 16]

   * global_leader can then iterate through the all_max_temps vector, matching with the socket and core from the order vectors
   * once all_num_values[i] is reached, a newline is created and a new node begins

   * the only remaining problem is getting the node name on global_leader
   */
  if (global_rank == global_leader) {
  writeSensorData(all_max_temps, all_socket_orders, all_core_orders, num_values);
  }
  MPI_Comm_free(&node_comm);
}

} // namespace sensors
