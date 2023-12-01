include <stdio.h>
#include <stdlib.h>
#include <string.h>  // Include this for the strstr function
#include "advection.h"

double readInputFile(char *fileName, char* tag) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        return -1;
    }

    char line[100];
    double value;

    while (fgets(line, 100, fp)) {
        if (strstr(line, tag)) {
            if (fgets(line, sizeof(line), fp)) {
                // Extract the value from the next line
                sscanf(line, "%lf", &value);
            }

            break;
        }
    }

    if (value == -1.0) {
        printf("Error: Tag '%s' not found in the input file.\n", tag);
    } else {
        printf("%lf\n", value);
    }

    fclose(fp);
    return value;
}

int main(int argc, char *argv[]) {
    double tstart = readInputFile(argv[1], "TSART");
    double tend   = readInputFile(argv[1], "TEND");
    double dt     = readInputFile(argv[1], "DT");
    double time   = 0.0;
    int Noutput   = readInputFile(argv[1], "OUTPUT_FREQUENCY");

    printf("%lf %lf %lf %lf %d\n", tstart, tend, dt, time, Noutput);

    return 0;
}
