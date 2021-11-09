#ifndef CALIBRATE_H
#define CALIBRATE_H

#include "TFile.h"

int main(int argc, char **argv);
void InitGRSISort();
void PrintUsage(char* argv[]);

std::string grsi_path;
TFile* source_file;
TFile* bg_file;

#endif
