#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <sched.h> // only works on linux

#include "freq.h"

namespace freq {

int readCPUFrequency() {
    int cpu = sched_getcpu();
    if (cpu < 0) {
        std::cerr << "Warning: Could not determine current CPU." << std::endl;
        return -1;
    }

    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE* fp = fopen(path, "r");
    if (!fp) {
        std::cerr << "Warning: Could not open file " << path << std::endl;
        return -1;
    }

    int cpu_freq = -1;
    if (fscanf(fp, "%d", &cpu_freq) != 1) {
        std::cerr << "Warning: Could not read frequency from " << path << std::endl;
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return cpu_freq;
}

} // end namespace freq
