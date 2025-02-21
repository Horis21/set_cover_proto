#include <stdio.h>
#include <time.h>

typedef struct {
    float result;
    int checks;
} ClassificationResult;

ClassificationResult classify(const float x[]);

int main() {
    int total_instances = 137;
    int features = 68;
    float dataset[total_instances][features + 1];

    FILE *file = fopen("hepatitis.csv", "r");
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
    int total_checks = 0;  // Track total number of checks

    for (int i = 0; i < total_instances; i++) {
        float features_only[features];
        for (int j = 0; j < features; j++) {
            features_only[j] = dataset[i][j + 1];
        }

        ClassificationResult result = classify(features_only);
        printf("Instance %d classified as: %.0f (Checks: %d)\n", i + 1, result.result, result.checks);

        if (result.result == dataset[i][0]) {
            correct_classifications++;
        }

        total_checks += result.checks;
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double avg_question_length = (double) total_checks / total_instances;

    printf("\nTotal correct classifications: %d out of %d\n", correct_classifications, total_instances);
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
    printf("Average question length: %.2f checks per instance\n", avg_question_length);

    return 0;
}

ClassificationResult classify(const float x[]) {
    int checks = 0;
    ClassificationResult result;

    if (++checks && x[34] <= 0.5) {
        if (++checks && x[0] <= 0.5) {
            if (++checks && x[36] <= 0.5) {
                result.result = 0.0f;
            } else {
                if (++checks && x[50] <= 0.5) {
                    result.result = 0.0f;
                } else {
                    if (++checks && x[10] <= 0.5) {
                        result.result = 1.0f;
                    } else {
                        if (++checks && x[66] <= 0.5) {
                            result.result = 0.0f;
                        } else {
                            result.result = 1.0f;
                        }
                    }
                }
            }
        } else {
            result.result = 1.0f;
        }
    } else {
        if (++checks && x[48] <= 0.5) {
            if (++checks && x[0] <= 0.5) {
                if (++checks && x[30] <= 0.5) {
                    if (++checks && x[8] <= 0.5) {
                        if (++checks && x[12] <= 0.5) {
                            result.result = 0.0f;
                        } else {
                            result.result = 1.0f;
                        }
                    } else {
                        result.result = 0.0f;
                    }
                } else {
                    if (++checks && x[16] <= 0.5) {
                        if (++checks && x[20] <= 0.5) {
                            result.result = 1.0f;
                        } else {
                            result.result = 0.0f;
                        }
                    } else {
                        if (++checks && x[6] <= 0.5) {
                            if (++checks && x[10] <= 0.5) {
                                result.result = 0.0f;
                            } else {
                                result.result = 1.0f;
                            }
                        } else {
                            result.result = 0.0f;
                        }
                    }
                }
            } else {
                result.result = 1.0f;
            }
        } else {
            if (++checks && x[4] <= 0.5) {
                if (++checks && x[66] <= 0.5) {
                    if (++checks && x[22] <= 0.5) {
                        if (++checks && x[26] <= 0.5) {
                            result.result = 0.0f;
                        } else {
                            result.result = 1.0f;
                        }
                    } else {
                        if (++checks && x[38] <= 0.5) {
                            result.result = 1.0f;
                        } else {
                            result.result = 0.0f;
                        }
                    }
                } else {
                    if (++checks && x[38] <= 0.5) {
                        result.result = 1.0f;
                    } else {
                        if (++checks && x[54] <= 0.5) {
                            result.result = 1.0f;
                        } else {
                            result.result = 0.0f;
                        }
                    }
                }
            } else {
                if (++checks && x[0] <= 0.5) {
                    result.result = 1.0f;
                } else {
                    if (++checks && x[20] <= 0.5) {
                        result.result = 1.0f;
                    } else {
                        if (++checks && x[40] <= 0.5) {
                            result.result = 1.0f;
                        } else {
                            result.result = 1.0f;
                        }
                    }
                }
            }
        }
    }

    result.checks = checks;
    return result;
}
