#include <stdio.h>
#include <time.h>

float classify(const float x[], int *check_count);

int main() {
    int total_instances = 122;
    int features = 15;

    float dataset[total_instances][features + 1];

    FILE *file = fopen("monk3_bin.csv", "r");
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

    int correct_classifications = 0;
    int total_checks = 0;

    for (int i = 0; i < total_instances; i++) {
        float features_only[features];
        for (int j = 0; j < features; j++) {
            features_only[j] = dataset[i][j + 1];
        }

        int check_count = 0;
        float result = classify(features_only, &check_count);
        printf("Instance %d classified as: %.0f (Checks used: %d)\n", i + 1, result, check_count);

        total_checks += check_count;

        if (result == dataset[i][0]) {
            correct_classifications++;
        }
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double avg_checks = (double)total_checks / total_instances;

    printf("\nTotal correct classifications: %d out of %d\n", correct_classifications, total_instances);
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
    printf("Average number of checks per classification: %.2f\n", avg_checks);

    return 0;
}

float classify(const float x[], int *check_count) {
    (*check_count)++; if (x[5] <= 0.5) {
        (*check_count)++; if (x[13] <= 0.5) {
            (*check_count)++; if (x[12] <= 0.5) {
                (*check_count)++; if (x[1] <= 0.5) {
                    return 1.0f;
                } else {
                    (*check_count)++; if (x[14] <= 0.5) {
                        (*check_count)++; if (x[10] <= 0.5) {
                            return 0.0f;
                        } else {
                            return 1.0f;
                        }
                    } else {
                        return 1.0f;
                    }
                }
            } else {
                (*check_count)++; if (x[6] <= 0.5) {
                    (*check_count)++; if (x[2] <= 0.5) {
                        (*check_count)++; if (x[14] <= 0.5) {
                            (*check_count)++; if (x[7] <= 0.5) {
                                return 0.0f;
                            } else {
                                (*check_count)++; if (x[3] <= 0.5) {
                                    return 1.0f;
                                } else {
                                    return 0.0f;
                                }
                            }
                        } else {
                            (*check_count)++; if (x[0] <= 0.5) {
                                (*check_count)++; if (x[3] <= 0.5) {
                                    (*check_count)++; if (x[7] <= 0.5) {
                                        return 0.0f;
                                    } else {
                                        return 1.0f;
                                    }
                                } else {
                                    return 1.0f;
                                }
                            } else {
                                return 1.0f;
                            }
                        }
                    } else {
                        return 1.0f;
                    }
                } else {
                    return 1.0f;
                }
            }
        } else {
            return 0.0f;
        }
    } else {
        (*check_count)++; if (x[6] <= 0.5) {
            (*check_count)++; if (x[0] <= 0.5) {
                return 1.0f;
            } else {
                return 0.0f;
            }
        } else {
            return 0.0f;
        }
    }
}
