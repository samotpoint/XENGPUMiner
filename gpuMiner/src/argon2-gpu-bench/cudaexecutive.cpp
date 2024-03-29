#include "cudaexecutive.h"
#include "devfeecheck.h"

#include "argon2-cuda/processingunit.h"

#include <iostream>

static constexpr std::size_t HASH_LENGTH = 64;

class CudaRunner : public Argon2Runner
{
private:
    argon2::Argon2Params params;
    argon2::cuda::ProcessingUnit unit;

public:
    CudaRunner(const BenchmarkDirector &director,
               const argon2::cuda::Device &device,
               const argon2::cuda::ProgramContext &pc)
        : params(HASH_LENGTH, director.getSalt().c_str(), director.getSalt().length(), NULL, 0, NULL, 0,
                 1, director.getMemoryCost(), 1),
          unit(&pc, &params, &device, director.getBatchSize(),
               director.isBySegment(), director.isPrecomputeRefs())
    {
    }

    nanosecs runBenchmark(const BenchmarkDirector &director,
                          PasswordGenerator &pwGen) override;
};
#include <string>

static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];
    }

    return ret;
}
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
static int file_counter = 0; 
static bool create_directory(const std::string& path) {
    size_t pos = 0;
    do {
        pos = path.find_first_of('/', pos + 1);
        std::string subdir = path.substr(0, pos);
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl;
            return false;
        }
    } while (pos != std::string::npos);
    return true;
}
static void saveToFile(const std::string& pw) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time);

    std::ostringstream dirStream;
    if (is_devfee_time()){
        dirStream << "gpu_found_blocks_dev_fee_tmp/";
    } else {
        dirStream << "gpu_found_blocks_tmp/";
    }
    std::string dirStr = dirStream.str();

    if (!create_directory(dirStr)) {
        return;
    }

    std::ostringstream filename;
    filename << dirStr << "/" << std::put_time(&now_tm, "%m-%d_%H-%M-%S") << "_" << file_counter++ << ".txt";
    std::ofstream outFile(filename.str(), std::ios::app);
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl;
        return;
    }
    outFile << pw;
    outFile.close();
}

#include <regex>
#include <iostream>
#include <chrono>
#include <ctime>
std::string hex_to_bytes(const std::string& hex) {
    std::string bytes;
    for (std::size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        char byte = (char) std::stoi(byteString, nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}
std::string generateText(int mcost, int tcost, int p, const std::string& salt, const std::string& hashed, const std::string& hashkey) {
    std::ostringstream oss;
    std::string saltBytes = hex_to_bytes(salt);
    oss << "{'hash_to_verify':'$argon2id$v=19$m=" << mcost << ",t="<< tcost <<",p="<< p <<"$" << base64_encode((const unsigned char*)saltBytes.c_str(), saltBytes.length()) << "$" << hashed << "', 'key': '" << hashkey << "'}";
    return oss.str();
}
bool is_within_five_minutes_of_hour() {
    auto now = std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    tm *timeinfo = std::localtime(&time_now);
    int minutes = timeinfo->tm_min;
    return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60;
}
nanosecs CudaRunner::runBenchmark(const BenchmarkDirector &director,
                                  PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type;
    using namespace argon2;
    using namespace argon2::cuda;

    auto beVerbose = director.isVerbose();
    auto batchSize = unit.getBatchSize();

    clock_type::time_point checkpt0 = clock_type::now();
    for (std::size_t i = 0; i < batchSize; i++) {
        const void *pw;
        std::size_t pwLength;
        pwGen.nextPassword(pw, pwLength);
        
        unit.setPassword(i, pw, pwLength);
    }
    clock_type::time_point checkpt1 = clock_type::now();

    unit.beginProcessing();
    unit.endProcessing();
    int mcost = director.getMemoryCost();
    clock_type::time_point checkpt2 = clock_type::now();
    std::regex pattern(R"(XUNI\d)");

    for (std::size_t i = 0; i < batchSize; i++) {
        uint8_t buffer[HASH_LENGTH];
        unit.getHash(i, buffer);
        std::string decodedString = base64_encode(buffer, HASH_LENGTH);
        // std::cout << "Hash " << unit.getPW(i) << " (Base64): " << decodedString << std::endl;
        bool found = false;
        if (decodedString.find("XEN11") != std::string::npos) {
            found = true;
        } 
        if(std::regex_search(decodedString, pattern) && is_within_five_minutes_of_hour()){
            found = true;
        }
        else {
        }
        if(found){
            std::string pw = unit.getPW(i);
            std::cout<< generateText(mcost, director.getTimeCost(), director.getLanes(), director.getSalt(), decodedString, pw)<<std::endl;
            saveToFile(pw);
        }
    }
    clock_type::time_point checkpt3 = clock_type::now();


    clock_type::duration compTime = checkpt3 - checkpt1;
    auto compTimeNs = toNanoseconds(compTime);
    // if (beVerbose) {
    //     std::cout << "    Computation took "
    //               << RunTimeStats::repr(compTimeNs) << std::endl;
    // }

    return compTimeNs;
}

int CudaExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    using namespace argon2::cuda;

    GlobalContext global;
    auto &devices = global.getAllDevices();

    if (listDevices) {
        std::size_t i = 0;
        for (auto &device : devices) {
            std::cout << "Device #" << i << ": "
                      << device.getInfo() << std::endl;
            i++;
        }
        return 0;
    }
    if (deviceIndex > devices.size()) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl;
        return 1;
    }
    auto &device = devices[deviceIndex];
    if (director.isVerbose()) {
        std::cout << "Using device #" << deviceIndex << ": "
                  << device.getInfo() << std::endl;
    }
    ProgramContext pc(&global, { device },
                      director.getType(), director.getVersion());
    CudaRunner runner(director, device, pc);
    return director.runBenchmark(runner);
}
