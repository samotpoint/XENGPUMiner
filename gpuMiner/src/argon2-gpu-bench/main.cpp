#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "benchmark.h"
#include "openclexecutive.h"
#include "cudaexecutive.h"
#include "cpuexecutive.h"

#include <iostream>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

bool is_devfee_time();

using namespace libcommandline;

struct Arguments
{
    std::string mode = "cuda";

    std::size_t deviceIndex = 0;

    std::string outputType = "ns";
    std::string outputMode = "verbose";

    std::size_t batchSize = 0;
    std::string kernelType = "oneshot";
    bool precomputeRefs = false;

    std::string benchmarkDeviceName = "unknowDevice";
    bool benchmark = false;
    
    bool showHelp = false;
    bool listDevices = false;
};

static CommandLineParser<Arguments> buildCmdLineParser()
{
    static const auto positional = PositionalArgumentHandler<Arguments>(
                [] (Arguments &, const std::string &) {});

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode; },
            "mode", 'm', "mode in which to run ('cuda' for CUDA, 'opencl' for OpenCL, or 'cpu' for CPU)", "cuda", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = index;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &name) { state.benchmarkDeviceName = name; state.benchmark = true; },
            "device-name", 't', "use device with name NAME", "unknowDevice", "NAME"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.outputType = type; },
            "output-type", 'o', "what to output (ns|ns-per-hash)", "ns", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.outputMode = mode; },
            "output-mode", '\0', "output mode (verbose|raw|mean|mean-and-mdev)", "verbose", "MODE"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.batchSize = num;
            }), "batch-size", 'b', "number of tasks per batch", "16", "N"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.kernelType = type; },
            "kernel-type", 'k', "kernel type (by-segment|oneshot)", "by-segment", "TYPE"),
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.precomputeRefs = true; },
            "precompute-refs", 'p', "precompute reference indices with Argon2i"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true; },
            "help", '?', "show this help and exit")
    };

    return CommandLineParser<Arguments>(
        "XENBlocks gpu miner: Only CUDA is supported in this version.",
        positional, options);
}

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include "shared.h"
#include <limits>

int difficulty = 90000;
std::mutex mtx;
void read_difficulty_periodically(const std::string& filename) {
    while (true) {
        std::ifstream file(filename);
        if (file.is_open()) {
            int new_difficulty;
            if (file >> new_difficulty) { // read difficulty
                std::lock_guard<std::mutex> lock(mtx);
                if(difficulty != new_difficulty){
                    difficulty = new_difficulty; // update difficulty
                    std::cout << "Updated difficulty to " << difficulty << std::endl;
                }
            }
            file.close(); 
        } else {
            std::cerr << "The local difficult.txt file was not recognized" << std::endl;
        }
        
        // sleep for 5 seconds
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

#include <atomic>
#include <csignal>
std::atomic<bool> running(true);
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    running = false;
    {
        std::lock_guard<std::mutex> lock(mtx);
        difficulty = difficulty - 1;
        std::cout << "change difficulty to " << difficulty << ", waiting process end" << std::endl;
    }
}

#include <cstdlib>
#include <regex>
#include <iomanip>
std::string getAccountValue(const std::string& configFilePath) {
    std::ifstream configFile(configFilePath);
    if (!configFile.is_open()) {
        std::cerr << "Error: Failed to open config file." << std::endl;
        std::abort();
    }

    std::string line;
    std::regex reg(R"(account\s*=\s*(.+))");  // Regular expression to match the account line and capture the value

    while (std::getline(configFile, line)) {  // Read the file line by line
        std::smatch match;
        if (std::regex_search(line, match, reg)) {
            std::string matchString = std::string(match[1]);
            if (matchString.find(".") != std::string::npos) {
                // Found a dot, output error message and abort program
                std::cerr << "Error: Account contains a dot, not supported yet" << std::endl;
                std::abort();
            }
            if (matchString.substr(0, 2) == "0x" || matchString.substr(0, 2) == "0X") {
                // No dot, but found prefix "0x", so return substring without "0x"
                return matchString.substr(2);
            }
            return match[1].str();  // Return the account value once found
        }
    }

    std::cerr << "Error: Account value not found in config file." << std::endl;
    std::abort();
}

int main(int, const char * const *argv)
{
    difficulty = 90000;
    // register signal SIGINT and signal handler
    signal(SIGINT, signalHandler);

    CommandLineParser<Arguments> parser = buildCmdLineParser();

    Arguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }
    if(args.listDevices){
        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13, "24691E54aFafe2416a8252097C9Ca67557271475",
                1, 120, 1, 1,
                false, args.precomputeRefs, 20000000,
                args.outputMode, args.outputType);
        CudaExecutive exec(args.deviceIndex, args.listDevices);
        exec.runBenchmark(director);
        return 0;
    }
    std::ifstream file("difficulty.txt");
    if (file.is_open()) {
        int new_difficulty;
        if (file >> new_difficulty) { // read difficulty
            std::lock_guard<std::mutex> lock(mtx);
            if(difficulty != new_difficulty){
                difficulty = new_difficulty; // update difficulty
                std::cout << "Updated difficulty to " << difficulty << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "The local difficult.txt file was not recognized" << std::endl;
    }
    // start a thread to read difficulty from file
    std::thread t(read_difficulty_periodically, "difficulty.txt"); 
    t.detach(); // detach thread from main thread, so it can run independently
    std::string salt = getAccountValue("config.conf");
    std::cout<< "Using "<<salt<<" as salt"<<std::endl;
    for(int i = 0; i < std::numeric_limits<size_t>::max(); i++){
        if(!running)break;

        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Current difficulty: " << difficulty << std::endl;
        }
        int mcost = difficulty;
        int batchSize = args.batchSize;
        if(args.batchSize == 0){
        #if HAVE_CUDA
            cudaSetDevice(args.deviceIndex); // Set device by index
            size_t freeMemory, totalMemory;
            cudaMemGetInfo(&freeMemory, &totalMemory);
            batchSize = freeMemory / 1.01 / mcost / 1024;
            printf("using batchsize:%d\n", batchSize);
        #endif
        }

        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13, salt,
                1, mcost, 1, batchSize,
                false, args.precomputeRefs, std::numeric_limits<size_t>::max(),
                args.outputMode, args.outputType);
        CudaExecutive exec(args.deviceIndex, args.listDevices);
        exec.runBenchmark(director);
    }
    return 0;
}

