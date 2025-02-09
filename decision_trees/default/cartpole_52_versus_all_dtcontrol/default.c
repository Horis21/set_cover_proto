#include <stdio.h>

float classify(const float x[]);

int main() {
    float x[] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,0.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    float result = classify(x);
    return 0;
}

float classify(const float x[]) {
	if (x[8] <= 0.5) {
		if (x[26] <= 0.5) {
			if (x[16] <= 0.5) {
				if (x[17] <= 0.5) {
					return 0.0f;
				}
				else {
					if (x[27] <= 0.5) {
						return 0.0f;
					}
					else {
						return 1.0f;
					}

				}

			}
			else {
				if (x[28] <= 0.5) {
					if (x[15] <= 0.5) {
						return 0.0f;
					}
					else {
						if (x[29] <= 0.5) {
							if (x[14] <= 0.5) {
								return 0.0f;
							}
							else {
								if (x[30] <= 0.5) {
									if (x[13] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[31] <= 0.5) {
											if (x[12] <= 0.5) {
												return 0.0f;
											}
											else {
												if (x[32] <= 0.5) {
													if (x[11] <= 0.5) {
														return 0.0f;
													}
													else {
														if (x[33] <= 0.5) {
															if (x[10] <= 0.5) {
																return 0.0f;
															}
															else {
																if (x[34] <= 0.5) {
																	if (x[9] <= 0.5) {
																		return 0.0f;
																	}
																	else {
																		if (x[35] <= 0.5) {
																			return 0.0f;
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
				else {
					return 1.0f;
				}

			}

		}
		else {
			return 1.0f;
		}

	}
	else {
		return 1.0f;
	}

}