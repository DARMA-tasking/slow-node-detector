#ifdef VT_ENABLED
#include <vt/transport.h>
#endif

#ifndef vt_check_enabled
#define vt_check_enabled(test_option) 0
#endif

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace perf {

namespace {

std::string sanitizeBenchmarkName(std::string benchmark_name) {
    if (benchmark_name.empty()) {
        return "perf";
    }

    for (char& ch : benchmark_name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_' && ch != '-') {
            ch = '_';
        }
    }

    return benchmark_name;
}

} // namespace

void startMeasurements() {
#if vt_check_enabled( perf )
    vt::theContext()->getTask()->startMetrics();
#endif
}

void stopMeasurements(const std::string& benchmark_name) {
#if vt_check_enabled( perf )
    vt::theContext()->getTask()->stopMetrics();

    int rank = 0;
    int size = 1;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Barrier(MPI_COMM_WORLD);

    std::unordered_map< std::string, uint64_t > metrics = vt::theContext()->getTask()->getMetrics();

    std::vector<std::pair<std::string, uint64_t>> ordered_metrics(metrics.begin(), metrics.end());
    std::sort(
      ordered_metrics.begin(),
      ordered_metrics.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    std::ostringstream local_csv;
    for (const auto& [name, value] : ordered_metrics) {
        local_csv << rank << "," << processor_name << "," << name << "," << value << "\n";
    }

    const std::string local_payload = local_csv.str();
    const int local_bytes = static_cast<int>(local_payload.size());

    std::vector<int> recv_counts;
    if (rank == 0) {
        recv_counts.resize(size);
    }

    MPI_Gather(
      &local_bytes,
      1,
      MPI_INT,
      rank == 0 ? recv_counts.data() : nullptr,
      1,
      MPI_INT,
      0,
      MPI_COMM_WORLD
    );

    std::vector<int> displacements;
    std::vector<char> gathered_payload;
    if (rank == 0) {
        displacements.resize(size, 0);
        int total_bytes = 0;
        for (int i = 0; i < size; ++i) {
            displacements[i] = total_bytes;
            total_bytes += recv_counts[i];
        }
        gathered_payload.resize(total_bytes);
    }

    MPI_Gatherv(
      local_payload.data(),
      local_bytes,
      MPI_CHAR,
      rank == 0 ? gathered_payload.data() : nullptr,
      rank == 0 ? recv_counts.data() : nullptr,
      rank == 0 ? displacements.data() : nullptr,
      MPI_CHAR,
      0,
      MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::string const output_name = "perf_metrics_" + sanitizeBenchmarkName(benchmark_name) + ".csv";
        std::ofstream out(output_name, std::ios::out | std::ios::trunc);
        out << "rank,hostname,metric,value\n";
        out.write(gathered_payload.data(), static_cast<std::streamsize>(gathered_payload.size()));
    }
#endif
}

} // namespace perf
