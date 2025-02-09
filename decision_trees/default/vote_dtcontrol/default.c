#include <stdio.h>

float classify(const float x[]);

int main() {
    float x[] = {1.f,0.f,0.f,0.f,1.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f,1.f,0.f,0.f,1.f,0.f,0.f,1.f,0.f,1.f,0.f,0.f,0.f,1.f,0.f};
    float result = classify(x);
    return 0;
}

float classify(const float x[]) {
	if (x[10] <= 0.5) {
		if (x[15] <= 0.5) {
			if (x[8] <= 0.5) {
				if (x[35] <= 0.5) {
					return 1.0f;
				}
				else {
					if (x[6] <= 0.5) {
						return 1.0f;
					}
					else {
						return 0.0f;
					}

				}

			}
			else {
				if (x[26] <= 0.5) {
					if (x[30] <= 0.5) {
						return 1.0f;
					}
					else {
						if (x[0] <= 0.5) {
							return 0.0f;
						}
						else {
							return 1.0f;
						}

					}

				}
				else {
					return 0.0f;
				}

			}

		}
		else {
			if (x[27] <= 0.5) {
				if (x[3] <= 0.5) {
					return 1.0f;
				}
				else {
					if (x[6] <= 0.5) {
						if (x[46] <= 0.5) {
							return 1.0f;
						}
						else {
							if (x[33] <= 0.5) {
								return 1.0f;
							}
							else {
								if (x[0] <= 0.5) {
									if (x[42] <= 0.5) {
										if (x[36] <= 0.5) {
											return 1.0f;
										}
										else {
											if (x[30] <= 0.5) {
												if (x[21] <= 0.5) {
													if (x[39] <= 0.5) {
														return 1.0f;
													}
													else {
														return 1.0f;
													}

												}
												else {
													return 1.0f;
												}

											}
											else {
												if (x[18] <= 0.5) {
													if (x[39] <= 0.5) {
														return 1.0f;
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

									}
									else {
										if (x[30] <= 0.5) {
											return 1.0f;
										}
										else {
											return 1.0f;
										}

									}

								}
								else {
									if (x[30] <= 0.5) {
										return 1.0f;
									}
									else {
										return 1.0f;
									}

								}

							}

						}

					}
					else {
						if (x[42] <= 0.5) {
							return 1.0f;
						}
						else {
							return 0.0f;
						}

					}

				}

			}
			else {
				if (x[30] <= 0.5) {
					if (x[3] <= 0.5) {
						if (x[25] <= 0.5) {
							return 1.0f;
						}
						else {
							if (x[18] <= 0.5) {
								if (x[4] <= 0.5) {
									return 1.0f;
								}
								else {
									if (x[42] <= 0.5) {
										if (x[46] <= 0.5) {
											return 1.0f;
										}
										else {
											return 1.0f;
										}

									}
									else {
										if (x[46] <= 0.5) {
											return 1.0f;
										}
										else {
											if (x[33] <= 0.5) {
												return 1.0f;
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

					}
					else {
						if (x[25] <= 0.5) {
							return 1.0f;
						}
						else {
							return 1.0f;
						}

					}

				}
				else {
					if (x[46] <= 0.5) {
						if (x[5] <= 0.5) {
							if (x[36] <= 0.5) {
								return 1.0f;
							}
							else {
								if (x[42] <= 0.5) {
									if (x[3] <= 0.5) {
										if (x[43] <= 0.5) {
											return 1.0f;
										}
										else {
											return 1.0f;
										}

									}
									else {
										if (x[0] <= 0.5) {
											return 1.0f;
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
						if (x[37] <= 0.5) {
							if (x[39] <= 0.5) {
								return 1.0f;
							}
							else {
								if (x[4] <= 0.5) {
									if (x[3] <= 0.5) {
										return 1.0f;
									}
									else {
										if (x[33] <= 0.5) {
											return 1.0f;
										}
										else {
											if (x[44] <= 0.5) {
												return 1.0f;
											}
											else {
												return 1.0f;
											}

										}

									}

								}
								else {
									if (x[42] <= 0.5) {
										return 1.0f;
									}
									else {
										return 1.0f;
									}

								}

							}

						}
						else {
							return 1.0f;
						}

					}

				}

			}

		}

	}
	else {
		if (x[31] <= 0.5) {
			if (x[18] <= 0.5) {
				return 0.0f;
			}
			else {
				if (x[27] <= 0.5) {
					if (x[46] <= 0.5) {
						if (x[5] <= 0.5) {
							if (x[45] <= 0.5) {
								if (x[3] <= 0.5) {
									if (x[21] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[40] <= 0.5) {
											return 0.0f;
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
							else {
								if (x[0] <= 0.5) {
									return 0.0f;
								}
								else {
									if (x[42] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[13] <= 0.5) {
											return 0.0f;
										}
										else {
											if (x[15] <= 0.5) {
												if (x[30] <= 0.5) {
													return 0.0f;
												}
												else {
													if (x[3] <= 0.5) {
														if (x[34] <= 0.5) {
															return 0.0f;
														}
														else {
															if (x[37] <= 0.5) {
																return 0.0f;
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
												return 0.0f;
											}

										}

									}

								}

							}

						}
						else {
							return 0.0f;
						}

					}
					else {
						if (x[6] <= 0.5) {
							return 0.0f;
						}
						else {
							if (x[0] <= 0.5) {
								if (x[4] <= 0.5) {
									return 0.0f;
								}
								else {
									if (x[33] <= 0.5) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}
							else {
								if (x[30] <= 0.5) {
									return 0.0f;
								}
								else {
									if (x[24] <= 0.5) {
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
					if (x[6] <= 0.5) {
						return 1.0f;
					}
					else {
						if (x[4] <= 0.5) {
							if (x[46] <= 0.5) {
								if (x[15] <= 0.5) {
									if (x[42] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[3] <= 0.5) {
											if (x[45] <= 0.5) {
												return 0.0f;
											}
											else {
												return 0.0f;
											}

										}
										else {
											if (x[45] <= 0.5) {
												return 0.0f;
											}
											else {
												return 0.0f;
											}

										}

									}

								}
								else {
									return 0.0f;
								}

							}
							else {
								if (x[3] <= 0.5) {
									return 0.0f;
								}
								else {
									if (x[15] <= 0.5) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}

						}
						else {
							if (x[43] <= 0.5) {
								if (x[45] <= 0.5) {
									if (x[34] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[40] <= 0.5) {
											return 0.0f;
										}
										else {
											if (x[21] <= 0.5) {
												return 0.0f;
											}
											else {
												if (x[46] <= 0.5) {
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
									if (x[34] <= 0.5) {
										return 0.0f;
									}
									else {
										return 0.0f;
									}

								}

							}
							else {
								return 1.0f;
							}

						}

					}

				}

			}

		}
		else {
			if (x[6] <= 0.5) {
				if (x[18] <= 0.5) {
					return 0.0f;
				}
				else {
					return 1.0f;
				}

			}
			else {
				if (x[27] <= 0.5) {
					if (x[3] <= 0.5) {
						if (x[39] <= 0.5) {
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
					if (x[12] <= 0.5) {
						if (x[46] <= 0.5) {
							if (x[33] <= 0.5) {
								if (x[36] <= 0.5) {
									if (x[3] <= 0.5) {
										return 0.0f;
									}
									else {
										if (x[0] <= 0.5) {
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
							else {
								return 1.0f;
							}

						}
						else {
							return 0.0f;
						}

					}
					else {
						return 1.0f;
					}

				}

			}

		}

	}

}