package jvcl;

public class ConvolveNaive {

	int boundaryConditions;
	
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	
	ConvolveUnrolled cu;
	DimShifter ds;
	
	public ConvolveNaive(int boundaryConditions) {
		this.boundaryConditions = boundaryConditions;
		cu = new ConvolveUnrolled(boundaryConditions);
		ds = new DimShifter();
	}
	
	public ConvolveNaive() {
		this.boundaryConditions = ZERO_BOUNDARY;
		cu = new ConvolveUnrolled(boundaryConditions);
		ds = new DimShifter();
	}

	public double[] convolveFD(double[] vector, double[] kernel) {
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		int halfLength = (int)( (kernelLength - 1) / 2.0);
		double[] result;
		if (kernelLength == 3) {
			result = cu.convolve3(vector, kernel);
		} else if (kernelLength == 5) {
			result = cu.convolve5(vector, kernel);
		} else {
			int adjN;
			result = new double[vectorLength];
			for (int n = 0; n < vectorLength; n++) {
				if (n < halfLength || n >= vectorLength - halfLength) {
					kernelLoop:
						if (boundaryConditions == ZERO_BOUNDARY) break kernelLoop;
						for (int p = 0; p < kernelLength; p++) {
							adjN = n + (p - halfLength);
							if (adjN < 0) {
								if (boundaryConditions == MIRROR_BOUNDARY) {
									adjN = Math.abs(adjN);
									result[n] += vector[adjN]*kernel[p];
								} else if (boundaryConditions == PERIODIC_BOUNDARY) {
									adjN = vectorLength + adjN;
									result[n] += vector[adjN]*kernel[p];
								} else throw new RuntimeException("Exception in FD beginning boundaries");// BCs
							} // if outside boundary
							if (adjN >= vectorLength) {
								if (boundaryConditions == MIRROR_BOUNDARY) {
									adjN = 2*vectorLength - adjN - 1;
									result[n] += vector[adjN]*kernel[p];
								} else if (boundaryConditions == PERIODIC_BOUNDARY) {
									adjN = adjN - vectorLength;
									result[n] += vector[adjN]*kernel[p];
								} else throw new RuntimeException("Exception in FD ending boundaries");// BCs
							} // if outside boundary
						} // for p
					} else { // if boundary
						for (int p = 0; p < kernelLength; p++) {
							adjN = n + (p - halfLength);
							result[n+halfLength]  += vector[adjN]*kernel[p];
						}
					} 
				} // for n
		}
		return result;
	}
	
	public double[][] convolveFD(double[][] image, double[] kernel, int dim) {
		if (dim > 1) throw new RuntimeException("Invalid dim");
		if (dim == 0) image = ds.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolveFD(image[n], kernel);
		}
		if (dim == 0) {
			return ds.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public double[][] convolveFD(double[][] image, double[] kernel) {
		return convolveFD(image, kernel, 0);
	}
	
	public double[][][] convolveFD(double[][][] volume, double[] kernel, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				volume[x][y] = convolveFD(volume[x][y], kernel);
			}
		}
		return volume;
		
	}
	
	public double[][][] convolveFD(double[][][] volume, double[] kernel) {
		return convolveFD(volume, kernel, 0);
	}
	
	public double[][] convolveFD(double[][] image, double[][] kernel) { 
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int halfWidth = (int)( (kernelWidth - 1) / 2.0);
		int halfHeight = (int)( (kernelHeight - 1) / 2.0);
		double[][] result;
		if (kernelWidth == 3 && kernelHeight == 3) {
			result = cu.convolve3(image, kernel);
		} else if (kernelWidth == 5 && kernelHeight == 5) {
			result = cu.convolve5(image, kernel);
		} else {
			int adjX;
			int adjY;
			result = new double[imageWidth][imageHeight];
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					boundaryCheck:
					if (x < halfWidth || x > imageWidth - halfWidth || y < halfHeight || y > imageHeight - halfHeight) { // if boundary
						if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
						for (int p = 0; p < kernelWidth; p++) {
							for (int q = 0; q < kernelHeight; q++) {
								adjX = x + (p - halfWidth);
								adjY = y + (q - halfHeight);
								if (adjX < 0) {
									if (boundaryConditions == MIRROR_BOUNDARY) {
										adjX = Math.abs(adjX);
									} else if (boundaryConditions == PERIODIC_BOUNDARY) {
										adjX = imageWidth + adjX;
									} else throw new RuntimeException("Exception in FD boundaries");// BCs
								} else if (adjX >= imageWidth) {
									if (boundaryConditions == MIRROR_BOUNDARY) {
										adjX = 2*imageWidth - adjX - 1;
									} else if (boundaryConditions == PERIODIC_BOUNDARY) {
										adjX = adjX - imageWidth;
									} else throw new RuntimeException("Exception in FD boundaries");// BCs
								}
								if (adjY < 0) {
									if (boundaryConditions == MIRROR_BOUNDARY) {
										adjY = Math.abs(adjY);
									} else if (boundaryConditions == PERIODIC_BOUNDARY) {
										adjY = imageWidth + adjY;
									} else throw new RuntimeException("Exception in FD boundaries");// BCs
								} else if (adjY >= imageHeight) {
									if (boundaryConditions == MIRROR_BOUNDARY) {
										adjY = 2*imageHeight - adjY - 1;
									} else if (boundaryConditions == PERIODIC_BOUNDARY) {
										adjY = adjY - imageHeight;
									} else throw new RuntimeException("Exception in FD boundaries");// BCs
								}
								result[x][y] += image[adjX][adjY]*kernel[p][q];
							} // for q
						} // for p
					} else { // if NOT boundary
						for (int p = 0; p < kernelWidth; p++) {
							for (int q = 0; q < kernelHeight; q++) {
								adjX = x + (p - halfWidth);
								adjY = y + (q - halfHeight);
								result[x][y] += image[adjX][adjY]*kernel[p][q];
							} // for q
						} // for p
					} // if boundary
				} // for y
			} // for x
		}
		return result;
	}
	
	public double[][][] convolveFD(double[][][] volume, double[][] kernel, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		
		for (int x = 0; x < volumeWidth; x++) {
				volume[x] = convolveFD(volume[x], kernel);
		}
		return volume;
	}
	
	public double[][][] convolveFD(double[][][] volume, double[][] kernel) {
		return convolveFD(volume, kernel, 0);
	}
	
	public double[][][] convolveFD(double[][][] volume, double[][][] kernel, boolean test) {
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		int volumeDepth = volume[0][0].length;
		int halfWidth = (int)( (kernelWidth - 1) / 2.0 );
		int halfHeight = (int)( (kernelHeight - 1) / 2.0);
		int halfDepth = (int)( (kernelDepth - 1) / 2.0 );
		double[][][] result;
		if (kernelHeight == 3 && kernelWidth == 3 && kernelDepth ==3 && !test) {
			result = cu.convolve3(volume, kernel);
		} else if (kernelHeight == 5 && kernelWidth == 5 && kernelDepth == 5 && !test) {
			result = cu.convolve5(volume, kernel);
		} else {
			int adjX, adjY, adjZ;
			result = new double[volumeWidth][volumeHeight][volumeDepth];
			for (int x = 0; x < volumeWidth; x++) {
				for (int y = 0; y < volumeHeight; y++) {
					for (int z = 0; z < volumeDepth; z++) {
						boundaryCheck:
						if (  x < halfWidth || x >= volumeWidth - halfWidth || y < halfHeight || y >= volumeHeight - halfHeight || 
								z < halfDepth || z >= volumeDepth - halfDepth ) {
							if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
							for (int p = 0; p < kernelWidth; p++) {
								for (int q = 0; q < kernelHeight; q++) {
									for (int r = 0; r < kernelDepth; r++) {
										adjX = x + (p - halfWidth);
										adjY = y + (q - halfHeight);
										adjZ = z + (r - halfDepth);
										if (adjX < 0) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjX = Math.abs(adjX);
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjX = volumeWidth + adjX;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										} else if (adjX >= volumeWidth) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjX = 2*volumeWidth - adjX - 1;
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjX = adjX - volumeWidth;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										}
										if (adjY < 0) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjY = Math.abs(adjY);
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjY = volumeWidth + adjY;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										} else if (adjY >= volumeHeight) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjY = 2*volumeHeight - adjY - 1;
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjY = adjY - volumeHeight;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										}
										if (adjZ < 0) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjZ = Math.abs(adjZ);
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjZ = volumeDepth + adjZ;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										} else if (adjZ >= volumeDepth) {
											if (boundaryConditions == MIRROR_BOUNDARY) {
												adjZ = 2*volumeDepth - adjZ - 1;
											} else if (boundaryConditions == PERIODIC_BOUNDARY) {
												adjZ= adjZ - volumeDepth;
											} else throw new RuntimeException("Exception in FD boundaries");// BCs
										}
										result[x][y][z] += volume[adjX][adjY][adjZ]*kernel[p][q][r];
									} // for r
								} // for q
							} // for p
						} else {
							for (int p = 0; p < kernelHeight; p++) {
								for (int q = 0; q < kernelWidth; q++) {
									for (int r = 0; r < kernelDepth; r++) {
										adjX = x + (p-halfHeight);
										adjY = y + (q-halfWidth);
										adjZ = z + (r-halfDepth);
										result[x][y][z] += volume[adjX][adjY][adjZ] * kernel[p][q][r];
									} // for r
								} // for q
							} // for p
						} // else 
					} // for z
				} // for y
			} // for x
		}
		return result;
	}
	
	
	
}
