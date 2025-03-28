#ifndef SRC_FREQ_H
#define SRC_FREQ_H

#include <vector>
#include <string>

namespace freq {

std::vector<int> readCPUFrequencies(const std::string& proc_name);

} // namespace freq

#endif // SRC_FREQ_H