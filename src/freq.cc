#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

#include "freq.h"

namespace freq {

std::vector<int> readCPUFrequencies() {
    std::vector<int> freqs;
    FILE* pipe = popen("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq", "r");
    if (!pipe) {
        std::cerr << "Warning: Could not read CPU frequencies." << std::endl;
        return freqs;
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::stringstream ss(buffer);
        int freq;
        while (ss >> freq) {
            freqs.push_back(freq);
        }
    }

    pclose(pipe);
    return freqs;
}

} // end namespace freq
