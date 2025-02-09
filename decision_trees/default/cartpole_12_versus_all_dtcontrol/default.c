#include <stdio.h>

float classify(const float x[]);

int main() {
    float x[] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float result = classify(x);
    return 0;
}

float classify(const float x[]) {
	if (x[25] <= 0.5) {
		return 0.0f;
	}
	else {
		if (x[9] <= 0.5) {
			return 0.0f;
		}
		else {
			if (x[2] <= 0.5) {
				if (x[24] <= 0.5) {
					return 0.0f;
				}
				else {
					if (x[3] <= 0.5) {
						if (x[23] <= 0.5) {
							return 0.0f;
						}
						else {
							if (x[4] <= 0.5) {
								if (x[22] <= 0.5) {
									return 0.0f;
								}
								else {
									if (x[5] <= 0.5) {
										if (x[21] <= 0.5) {
											return 0.0f;
										}
										else {
											if (x[6] <= 0.5) {
												if (x[20] <= 0.5) {
													return 0.0f;
												}
												else {
													if (x[7] <= 0.5) {
														if (x[19] <= 0.5) {
															return 0.0f;
														}
														else {
															if (x[8] <= 0.5) {
																if (x[18] <= 0.5) {
																	return 0.0f;
																}
																else {
																	return 1.0f;
																}

															}
															else {
																return 1.0f;
															}

														}

													}
													else {
														return 1.0f;
													}

												}

											}
											else {
												return 1.0f;
											}

										}

									}
									else {
										return 1.0f;
									}

								}

							}
							else {
								return 1.0f;
							}

						}

					}
					else {
						return 1.0f;
					}

				}

			}
			else {
				return 1.0f;
			}

		}

	}

}