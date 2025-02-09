#include <stdio.h>
#include <time.h>
float classify(const float x[]);

int main() {
    // Define dataset dimensions
    int total_instances = 137;  // Number of instances in the dataset
    int features = 68;          // Number of features (excluding the label column)

    // Array to hold the dataset (including labels)
    float dataset[total_instances][features + 1];  // +1 for the label column

    // Open the dataset file
    FILE *file = fopen("hepatitis.csv", "r");
    if (file == NULL) {
        printf("Error opening the dataset file.\n");
        return -1;
    }

    // Read the dataset from the file
    for (int i = 0; i < total_instances; i++) {
        for (int j = 0; j < features + 1; j++) {
            if (fscanf(file, "%f", &dataset[i][j]) != 1) {
                printf("Error reading data from file.\n");
                fclose(file);
                return -1;
            }
        }
    }

    // Close the file after reading
    fclose(file);

    // Start measuring time
    clock_t start_time = clock();

    int correct_classifications = 0;

    // Loop through each instance (row)
    for (int i = 0; i < total_instances; i++) {
        // Extract features (excluding the label)
        float features_only[features];
        for (int j = 0; j < features; j++) {
            features_only[j] = dataset[i][j + 1];  // Skip the first column (label)
        }

        // Classify the instance
        float result = classify(features_only);
        printf("Instance %d classified as: %.0f\n", i + 1, result);

        // Assuming the label is in the first column, compare the predicted result with the actual label
        if (result == dataset[i][0]) {
            correct_classifications++;
        }
    }

    // End measuring time
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Output the classification result and elapsed time
    printf("\nTotal correct classifications: %d out of %d\n", correct_classifications, total_instances);
    printf("Elapsed time: %.6f seconds\n", elapsed_time);

    return 0;
}

float classify(const float x[]) {
	if (x[34] <= 0.5) {
		if (x[0] <= 0.5) {
			if (x[36] <= 0.5) {
				return 0.0f;
			}
			else {
				if (x[50] <= 0.5) {
					return 0.0f;
				}
				else {
					if (x[10] <= 0.5) {
						return 1.0f;
					}
					else {
						if (x[66] <= 0.5) {
							return 0.0f;
						}
						else {
							return 1.0f;
						}

					}

				}

			}

		}
		else {
			return 1.0f;
		}

	}
	else {
		if (x[48] <= 0.5) {
			if (x[0] <= 0.5) {
				if (x[30] <= 0.5) {
					if (x[8] <= 0.5) {
						if (x[12] <= 0.5) {
							return 0.0f;
						}
						else {
							return 1.0f;
						}

					}
					else {
						return 0.0f;
					}

				}
				else {
					if (x[16] <= 0.5) {
						if (x[20] <= 0.5) {
							return 1.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[6] <= 0.5) {
							if (x[10] <= 0.5) {
								return 0.0f;
							}
							else {
								return 1.0f;
							}

						}
						else {
							return 0.0f;
						}

					}

				}

			}
			else {
				return 1.0f;
			}

		}
		else {
			if (x[4] <= 0.5) {
				if (x[66] <= 0.5) {
					if (x[22] <= 0.5) {
						if (x[26] <= 0.5) {
							return 0.0f;
						}
						else {
							return 1.0f;
						}

					}
					else {
						if (x[38] <= 0.5) {
							return 1.0f;
						}
						else {
							return 0.0f;
						}

					}

				}
				else {
					if (x[38] <= 0.5) {
						return 1.0f;
					}
					else {
						if (x[54] <= 0.5) {
							return 1.0f;
						}
						else {
							return 0.0f;
						}

					}

				}

			}
			else {
				if (x[0] <= 0.5) {
					return 1.0f;
				}
				else {
					if (x[20] <= 0.5) {
						return 1.0f;
					}
					else {
						if (x[40] <= 0.5) {
							return 1.0f;
						}
						else {
							return 1.0f;
						}

					}

				}

			}

		}

	}

}
