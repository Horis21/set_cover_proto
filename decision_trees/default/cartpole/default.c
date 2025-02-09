#include <stdio.h>

float classify(const float x[]);

int main() {
    float x[] = {3.12f,-0.9f};
    float result = classify(x);
    return 0;
}

float classify(const float x[]) {
	if (x[0] <= 3.1600000858306885) {
		if (x[1] <= 0.75) {
			if (x[1] <= -0.44999998807907104) {
				if (x[1] <= -0.6499999761581421) {
					if (x[0] <= 3.0799999237060547) {
						if (x[0] <= 3.0) {
							return 2.4f;
						}
						else {
							if (x[1] <= -0.75) {
								return 2.3f;
							}
							else {
								return 0.2f;
							}

						}

					}
					else {
						if (x[1] <= -0.8500000238418579) {
							return 2.2f;
						}
						else {
							if (x[1] <= -0.75) {
								return 0.1f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}
				else {
					if (x[0] <= 2.9200000762939453) {
						if (x[0] <= 2.8399999141693115) {
							return 2.7f;
						}
						else {
							if (x[1] <= -0.550000011920929) {
								return 2.5f;
							}
							else {
								return 0.4f;
							}

						}

					}
					else {
						if (x[0] <= 3.0) {
							if (x[1] <= -0.550000011920929) {
								return 0.3f;
							}
							else {
								return 0.0f;
							}

						}
						else {
							if (x[0] <= 3.0799999237060547) {
								if (x[1] <= -0.550000011920929) {
									return 0.0f;
								}
								else {
									return 0.0f;
								}

							}
							else {
								if (x[1] <= -0.550000011920929) {
									return 0.0f;
								}
								else {
									return 0.0f;
								}

							}

						}

					}

				}

			}
			else {
				if (x[1] <= 0.15000000596046448) {
					if (x[0] <= 2.759999990463257) {
						if (x[0] <= 2.5999999046325684) {
							if (x[1] <= -0.05000000074505806) {
								if (x[0] <= 2.5199999809265137) {
									return 3.6f;
								}
								else {
									if (x[1] <= -0.15000000596046448) {
										return 3.3f;
									}
									else {
										return 0.8f;
									}

								}

							}
							else {
								if (x[1] <= 0.05000000074505806) {
									if (x[0] <= 2.440000057220459) {
										return 3.9f;
									}
									else {
										if (x[0] <= 2.5199999809265137) {
											return 0.9f;
										}
										else {
											return 0.0f;
										}

									}

								}
								else {
									if (x[0] <= 2.5199999809265137) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}

						}
						else {
							if (x[1] <= -0.25) {
								if (x[0] <= 2.680000066757202) {
									return 3.1f;
								}
								else {
									if (x[1] <= -0.3499999940395355) {
										return 2.9f;
									}
									else {
										return 0.6f;
									}

								}

							}
							else {
								if (x[1] <= -0.05000000074505806) {
									if (x[0] <= 2.680000066757202) {
										if (x[1] <= -0.15000000596046448) {
											return 0.7f;
										}
										else {
											return 0.0f;
										}

									}
									else {
										if (x[1] <= -0.15000000596046448) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}

								}
								else {
									if (x[0] <= 2.680000066757202) {
										if (x[1] <= 0.05000000074505806) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}
									else {
										return 0.0f;
									}

								}

							}

						}

					}
					else {
						if (x[1] <= -0.15000000596046448) {
							if (x[0] <= 2.9200000762939453) {
								if (x[1] <= -0.3499999940395355) {
									if (x[0] <= 2.8399999141693115) {
										return 0.5f;
									}
									else {
										return 0.0f;
									}

								}
								else {
									if (x[0] <= 2.8399999141693115) {
										if (x[1] <= -0.25) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}
									else {
										if (x[1] <= -0.25) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}

								}

							}
							else {
								if (x[0] <= 3.0) {
									if (x[1] <= -0.3499999940395355) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}
								else {
									return 0.0f;
								}

							}

						}
						else {
							return 0.0f;
						}

					}

				}
				else {
					if (x[0] <= 3.0) {
						return 0.0f;
					}
					else {
						if (x[1] <= 0.6499999761581421) {
							if (x[1] <= 0.550000011920929) {
								return 0.0f;
							}
							else {
								if (x[0] <= 3.0799999237060547) {
									return 0.0f;
								}
								else {
									return 0.0f;
								}

							}

						}
						else {
							if (x[0] <= 3.0799999237060547) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}

			}

		}
		else {
			if (x[0] <= 2.759999990463257) {
				if (x[1] <= 0.8500000238418579) {
					if (x[0] <= 2.5199999809265137) {
						if (x[0] <= 2.440000057220459) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 2.5999999046325684) {
							return 0.0f;
						}
						else {
							if (x[0] <= 2.680000066757202) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}
				else {
					if (x[0] <= 2.5199999809265137) {
						if (x[0] <= 2.440000057220459) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 2.5999999046325684) {
							return 0.0f;
						}
						else {
							if (x[0] <= 2.680000066757202) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}

			}
			else {
				if (x[1] <= 0.8500000238418579) {
					if (x[0] <= 2.9200000762939453) {
						if (x[0] <= 2.8399999141693115) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 3.0) {
							return 0.0f;
						}
						else {
							if (x[0] <= 3.0799999237060547) {
								return 0.0f;
							}
							else {
								return -0.1f;
							}

						}

					}

				}
				else {
					if (x[0] <= 2.9200000762939453) {
						return 0.0f;
					}
					else {
						if (x[0] <= 3.0) {
							return 0.0f;
						}
						else {
							if (x[0] <= 3.0799999237060547) {
								return 0.0f;
							}
							else {
								return -2.2f;
							}

						}

					}

				}

			}

		}

	}
	else {
		if (x[1] <= -0.75) {
			if (x[1] <= -0.8500000238418579) {
				if (x[0] <= 3.4800000190734863) {
					if (x[0] <= 3.319999933242798) {
						return 0.0f;
					}
					else {
						if (x[0] <= 3.4000000953674316) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}

				}
				else {
					if (x[0] <= 3.640000104904175) {
						if (x[0] <= 3.559999942779541) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 3.7200000286102295) {
							return 0.0f;
						}
						else {
							if (x[0] <= 3.799999952316284) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}

			}
			else {
				if (x[0] <= 3.4800000190734863) {
					if (x[0] <= 3.319999933242798) {
						if (x[0] <= 3.240000009536743) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 3.4000000953674316) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}

				}
				else {
					if (x[0] <= 3.640000104904175) {
						if (x[0] <= 3.559999942779541) {
							return 0.0f;
						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[0] <= 3.7200000286102295) {
							return 0.0f;
						}
						else {
							if (x[0] <= 3.799999952316284) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}

					}

				}

			}

		}
		else {
			if (x[1] <= 0.3499999940395355) {
				if (x[1] <= -0.15000000596046448) {
					if (x[1] <= -0.6499999761581421) {
						if (x[0] <= 3.319999933242798) {
							if (x[0] <= 3.240000009536743) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}
						else {
							return 0.0f;
						}

					}
					else {
						return 0.0f;
					}

				}
				else {
					if (x[0] <= 3.640000104904175) {
						if (x[0] <= 3.4800000190734863) {
							if (x[1] <= 0.15000000596046448) {
								return 0.0f;
							}
							else {
								if (x[0] <= 3.4000000953674316) {
									if (x[0] <= 3.319999933242798) {
										return 0.0f;
									}
									else {
										if (x[1] <= 0.25) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}

								}
								else {
									if (x[1] <= 0.25) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}

						}
						else {
							if (x[1] <= 0.05000000074505806) {
								if (x[0] <= 3.559999942779541) {
									return 0.0f;
								}
								else {
									if (x[1] <= -0.05000000074505806) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}
							else {
								if (x[1] <= 0.25) {
									if (x[0] <= 3.559999942779541) {
										if (x[1] <= 0.15000000596046448) {
											return 0.0f;
										}
										else {
											return 0.0f;
										}

									}
									else {
										if (x[1] <= 0.15000000596046448) {
											return 0.0f;
										}
										else {
											return -0.6f;
										}

									}

								}
								else {
									if (x[0] <= 3.559999942779541) {
										return -0.5f;
									}
									else {
										return -3.0f;
									}

								}

							}

						}

					}
					else {
						if (x[1] <= 0.05000000074505806) {
							if (x[1] <= -0.05000000074505806) {
								if (x[0] <= 3.7200000286102295) {
									return 0.0f;
								}
								else {
									return 0.0f;
								}

							}
							else {
								if (x[0] <= 3.7200000286102295) {
									return 0.0f;
								}
								else {
									if (x[0] <= 3.799999952316284) {
										return -0.9f;
									}
									else {
										return -3.7f;
									}

								}

							}

						}
						else {
							if (x[0] <= 3.7200000286102295) {
								if (x[1] <= 0.15000000596046448) {
									return -0.7f;
								}
								else {
									return -3.2f;
								}

							}
							else {
								return -3.4f;
							}

						}

					}

				}

			}
			else {
				if (x[1] <= 0.550000011920929) {
					if (x[0] <= 3.4000000953674316) {
						if (x[0] <= 3.240000009536743) {
							if (x[1] <= 0.44999998807907104) {
								return 0.0f;
							}
							else {
								return 0.0f;
							}

						}
						else {
							if (x[0] <= 3.319999933242798) {
								if (x[1] <= 0.44999998807907104) {
									return 0.0f;
								}
								else {
									return 0.0f;
								}

							}
							else {
								if (x[1] <= 0.44999998807907104) {
									return 0.0f;
								}
								else {
									return -0.3f;
								}

							}

						}

					}
					else {
						if (x[0] <= 3.4800000190734863) {
							if (x[1] <= 0.44999998807907104) {
								return -0.4f;
							}
							else {
								return -2.6f;
							}

						}
						else {
							return -2.8f;
						}

					}

				}
				else {
					if (x[0] <= 3.240000009536743) {
						if (x[1] <= 0.6499999761581421) {
							return 0.0f;
						}
						else {
							if (x[1] <= 0.75) {
								return -0.2f;
							}
							else {
								return -2.2f;
							}

						}

					}
					else {
						if (x[0] <= 3.319999933242798) {
							if (x[1] <= 0.6499999761581421) {
								return -0.2f;
							}
							else {
								return -2.4f;
							}

						}
						else {
							return -2.5f;
						}

					}

				}

			}

		}

	}

}