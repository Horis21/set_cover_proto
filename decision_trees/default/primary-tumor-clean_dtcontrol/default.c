#include <stdio.h>
#include <time.h>

float classify(const float x[], int *check_count);

int main() {
    int total_instances = 283;
    int features = 31;
    float dataset[total_instances][features + 1];

    FILE *file = fopen("primary-tumor-clean.csv", "r");
    if (file == NULL) {
        printf("Error opening the dataset file.\n");
        return -1;
    }

    for (int i = 0; i < total_instances; i++) {
        for (int j = 0; j < features + 1; j++) {
            if (fscanf(file, "%f", &dataset[i][j]) != 1) {
                printf("Error reading data from file.\n");
                fclose(file);
                return -1;
            }
        }
    }
    fclose(file);

    clock_t start_time = clock();

    int total_checks = 0;  // Sum of all checks
    int correct_classifications = 0;

    for (int i = 0; i < total_instances; i++) {
        float features_only[features];
        for (int j = 0; j < features; j++) {
            features_only[j] = dataset[i][j + 1];
        }

        int check_count = 0; // Track checks per instance
        float result = classify(features_only, &check_count);
        printf("Instance %d classified as: %.0f (Checks used: %d)\n", i + 1, result, check_count);

        total_checks += check_count; // Sum all checks

        if (result == dataset[i][0]) {
            correct_classifications++;
        }
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double avg_checks = (double)total_checks / total_instances;

    printf("\nTotal correct classifications: %d out of %d\n", correct_classifications, total_instances);
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
    printf("Average checks per classification: %.2f\n", avg_checks);

    return 0;
}

float classify(const float x[], int *check_count) {
    (*check_count)++; if (x[27] <= 0.5) {
        (*check_count)++; if (x[13] <= 0.5) {
            (*check_count)++; if (x[29] <= 0.5) {
                (*check_count)++; if (x[19] <= 0.5) {
                    (*check_count)++; if (x[21] <= 0.5) {
                        (*check_count)++; if (x[11] <= 0.5) {
                            (*check_count)++; if (x[5] <= 0.5) {
                                (*check_count)++; if (x[3] <= 0.5) {
                                    (*check_count)++; if (x[9] <= 0.5) {
                                        (*check_count)++; if (x[25] <= 0.5) {
                                            return 0.0f;
                                        }
                                        (*check_count)++;
                                        return 0.0f;
                                    }
                                    (*check_count)++;
                                    return 0.0f;
                                }
                                (*check_count)++;
                                return 0.0f;
                            }
                            (*check_count)++;
                            return 0.0f;
                        }
                        (*check_count)++;
                        return 0.0f;
                    }
                    (*check_count)++;
                    return 0.0f;
                }
                (*check_count)++;
                return 0.0f;
            }
            (*check_count)++;
            return 0.0f;
        }
        (*check_count)++;
        return 0.0f;
    }
    (*check_count)++;
    return 0.0f;
}
