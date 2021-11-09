//////////////////////////////////////////////////////////////////////////////////
// Calibrates energy based on bg peaks; run by run
//
// Author:        Connor Natzke (cnatzke@triumf.ca)
// Creation Date: 08-11-2021
// Last Update:  08-11-2021
// Usage:
//
//////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TGRSIUtilities.h"
#include "TParserLibrary.h"
#include "TEnv.h"

#include "Calibrate.h"

int main(int argc, char **argv)
{
    if (argc == 1) { // no inputs given
        PrintUsage(argv);
        return 0;
    }
    else if (argc == 2) {
        std::cout << "Histograms written to: " << argv[1] << std::endl;
    }
    else if (argc == 3) {
        grsi_path = getenv("GRSISYS");
        if(grsi_path.length() > 0) {
            grsi_path += "/";
        }
        grsi_path += ".grsirc";
        gEnv->ReadFile(grsi_path.c_str(), kEnvChange);

        TParserLibrary::Get()->Load();
    //    InitGRSISort();
        // makes output look nicer
        std::cout << std::endl;
    }

    return 0;
} // main()

void InitGRSISort(){
    // makes time retrival happy and loads GRSIEnv
    grsi_path = getenv("GRSISYS");
    if(grsi_path.length() > 0) {
        grsi_path += "/";
    }
    grsi_path += ".grsirc";
    gEnv->ReadFile(grsi_path.c_str(), kEnvChange);

    TParserLibrary::Get()->Load();
} // end InitGRSISort

/******************************************************************************
 * Prints usage message and version
 *****************************************************************************/
void PrintUsage(char* argv[]){
//    std::cerr << argv[0] << " Version: " << Calibrate_VERSION_MAJOR << "." << Calibrate_VERSION_MINOR << "\n"
    std::cerr << argv[0] << " Version: " <<  "0.1.0\n"
              << "\n----- Background Subtractions ------\n"
              << "usage: " << argv[0] << " source_file background_file \n"
              << " source_file: Source histograms\n"
              << " background_file: Background histograms\n"
              << "\n----- Matrix Creation ------\n"
              << "usage: " << argv[0] << " histogram_file\n"
              << " histogram_file: ROOT file containing background subtracted histograms\n"
              << std::endl;
} // end PrintUsage
