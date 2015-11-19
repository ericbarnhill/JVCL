/* Copyright (c) 2015 Eric Barnhill
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/

package jvcl;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math4.complex.Complex;

public class FDCPUUnrolled {

	int boundaryConditions;
	
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	
	ExecutorService threadPool;
	int cores;
	
	/** Unrolled convolutions for small kernels. This produces a performance boost despite the claims that the JVM makes it unnecessary,
	* particularly when the convolution is fast enough that the JVM is still warming up. If you used the Autotuner the root Convolve class will
	* call these unrolled methods when they benefit you.
	* @param boundaryConditions 0=Zero boundaries, slightly faster 1=Mirror boundaries 2=Periodic boundaries
	*/
	public FDCPUUnrolled(int boundaryConditions) {
		this.boundaryConditions = boundaryConditions;
		 cores = Runtime.getRuntime().availableProcessors();
		 threadPool = Executors.newFixedThreadPool(cores);
	}
	
	/** Unrolled convolutions for small kernels, see {@link #FDCPUUnrolled(int boundaryConditions) previous constructor}. Default boundaries
	* are zero boundaries.
	*/
	public FDCPUUnrolled() {
		this.boundaryConditions = ZERO_BOUNDARY;
		cores = Runtime.getRuntime().availableProcessors();
		threadPool = Executors.newFixedThreadPool(cores);
	}
	
	class DTConvolver2d implements Runnable {
		int x;
		double[][] fPad, r;
		double[] g;
		public DTConvolver2d(double[][] fPad, double[] g, double[][] r, int x) {
			this.fPad = fPad;
			this.g = g;
			this.x = x;
			this.r = r;
		}
		public void run() {
			r[x] = convolve10(fPad[x], g, r[x]);
		}
	}
	
	class DTConvolver3d implements Runnable {
		int x, y;
		double[][][] fPad, r;
		double[] g;
		public DTConvolver3d(double[][][] fPad, double[] g, double[][][] r, int x, int y) {
			this.fPad = fPad;
			this.g = g;
			this.x = x;
		}
		public void run() {
			r[x][y] = convolve10(fPad[x][y], g, r[x][y]);
		}
	}

	
	public double[] convolve(double[] vector, double[] kernel) {
		if (kernel.length == 3) {
			return convolve3(vector, kernel);
		} else return convolve5(vector, kernel);
	}
	
	public double[][] convolve(double[][] vector, double[][] kernel) {
		if (kernel.length == 3) {
			return convolve3(vector, kernel);
		} else return convolve5(vector, kernel);
	}
	
	public double[][][] convolve(double[][][] vector, double[][][] kernel) {
		if (kernel.length == 3) {
			return convolve3(vector, kernel);
		} else return convolve5(vector, kernel);
	}
	
	public double[] convolve3(double[] vector, double[] kernel) {
		int vectorLength = vector.length;
		if (kernel.length != 3) {
			throw new RuntimeException("Incorrect call to convolve3");
		}
		double[] result = new double[vectorLength];
		for (int n = 1; n < vectorLength - 1; n++) {
			result[n+1] = vector[n-1]*kernel[0] +
			vector[n]*kernel[1] +
			vector[n+1]*kernel[2];
		}
		if (boundaryConditions == MIRROR_BOUNDARY) {
			result[0] = vector[1]*kernel[0]+vector[0]*kernel[1]+vector[1]*kernel[2];
			result[vectorLength-1] = vector[vectorLength-2]*kernel[0]+vector[vectorLength-1]*kernel[1]+vector[vectorLength-2]*kernel[2];
		} else if (boundaryConditions == PERIODIC_BOUNDARY) {
			result[0] = vector[vectorLength-1]*kernel[0]+vector[0]*kernel[1]+vector[1]*kernel[2];
			result[vectorLength-1] = vector[vectorLength-2]*kernel[0]+vector[vectorLength-1]*kernel[1]+vector[0]*kernel[2];
		}
		return result;
	}
	
	public double[][] convolve3(double[][] image, double[][] kernel) {
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageArea = imageWidth*imageHeight;
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
		if (kernelHeight != 3 || kernelWidth != 3) {
			throw new RuntimeException("Incorrect Call To Convolve3");
		}
		double[][] result = new double[imageWidth][imageHeight];
		int x, y, p, q, pq, adjX, adjY;
		for (int xy = 0; xy < imageArea; xy++) {
			x = imageArea % imageWidth;
			y = xy / imageWidth;
			boundaryCheck:
			if (x < 1 || x >= imageWidth - 1 || y < 1 || y >= imageHeight - 1) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pq = 0; pq < 9; pq++) {
					p = pq % 3;
					q = pq / 3;
					adjX = x + (p - 2);
					adjY = y + (q - 2);
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
				} // for pq
			} else {
				result[x][y] = 
					image[x-1][y-1]*kernel[0][0] + 
					image[x-1][y+0]*kernel[0][1] + 
					image[x-1][y+1]*kernel[0][2] + 
					image[x+0][y-1]*kernel[1][0] + 
					image[x+0][y+0]*kernel[1][1] + 
					image[x+0][y+1]*kernel[1][2] + 
					image[x+1][y-1]*kernel[2][0] + 
					image[x+1][y+0]*kernel[2][1] + 
					image[x+1][y+1]*kernel[2][2];
			} // if boundary
		} // for xy
		return result;
	}

	public double[][][] convolve3(double[][][] volume, double[][][] kernel) {
		
		int volumeWidth = volume[0].length;
		int volumeHeight = volume.length;
		int volumeDepth = volume[0][0].length;
		int totalArea = volumeWidth*volumeHeight;
		int totalVolume = totalArea*volumeDepth;
		int kernelWidth = kernel[0].length;
		int kernelHeight = kernel.length;
		int kernelDepth = kernel[0][0].length;
		int kernelArea = kernelWidth*kernelHeight;
		if (kernelHeight != 3 || kernelWidth != 3 || kernelDepth != 3) {
			throw new RuntimeException("Incorrect Call To Convolve3");
		}
		double[][][] result = new double[volumeWidth][volumeHeight][volumeDepth];
		int x, y, z, p, q, r, pqr, adjX, adjY, adjZ;
		for (int xyz = 0; xyz < totalVolume; xyz++) {
			z = xyz / totalArea;
			y = (xyz - z*totalArea) / volumeWidth;
			x = xyz % volumeWidth;
			boundaryCheck:
			if ( x < 1 || x >= volumeWidth-1 || y < 1 || y >= volumeHeight-1 || 
					z < 1 || z >= volumeHeight-1 ) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pqr = 0; pqr < 27; pqr++) {
					r = pqr / 9;
					q = (pqr - r*9) / 3;
					p = pqr % 3;
					adjX = x + (p - 1);
					adjY = y + (q - 1);
					adjZ = z + (r - 1);
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
							adjZ = adjZ - volumeDepth;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y][z] += volume[adjX][adjY][adjZ]*kernel[p][q][r];
				} // for pqr
			} else { // if not boundary
				result[x][y][z] = 
					volume[x-1][y-1][z-1]*kernel[0][0][0] + 
					volume[x-1][y-1][z+0]*kernel[0][0][1] + 
					volume[x-1][y-1][z+1]*kernel[0][0][2] + 
					volume[x-1][y+0][z-1]*kernel[0][1][0] + 
					volume[x-1][y+0][z+0]*kernel[0][1][1] + 
					volume[x-1][y+0][z+1]*kernel[0][1][2] + 
					volume[x-1][y+1][z-1]*kernel[0][2][0] + 
					volume[x-1][y+1][z+0]*kernel[0][2][1] + 
					volume[x-1][y+1][z+1]*kernel[0][2][2] + 
					volume[x+0][y-1][z-1]*kernel[1][0][0] + 
					volume[x+0][y-1][z+0]*kernel[1][0][1] + 
					volume[x+0][y-1][z+1]*kernel[1][0][2] + 
					volume[x+0][y+0][z-1]*kernel[1][1][0] + 
					volume[x+0][y+0][z+0]*kernel[1][1][1] + 
					volume[x+0][y+0][z+1]*kernel[1][1][2] + 
					volume[x+0][y+1][z-1]*kernel[1][2][0] + 
					volume[x+0][y+1][z+0]*kernel[1][2][1] + 
					volume[x+0][y+1][z+1]*kernel[1][2][2] + 
					volume[x+1][y-1][z-1]*kernel[2][0][0] + 
					volume[x+1][y-1][z+0]*kernel[2][0][1] + 
					volume[x+1][y-1][z+1]*kernel[2][0][2] + 
					volume[x+1][y+0][z-1]*kernel[2][1][0] + 
					volume[x+1][y+0][z+0]*kernel[2][1][1] + 
					volume[x+1][y+0][z+1]*kernel[2][1][2] + 
					volume[x+1][y+1][z-1]*kernel[2][2][0] + 
					volume[x+1][y+1][z+0]*kernel[2][2][1] + 
					volume[x+1][y+1][z+1]*kernel[2][2][2];
			} // if not boundary
		} // for xyz
		return result;
	}
	
	public double[] convolve5(double[] vector, double[] kernel) {
		int vectorLength = vector.length;
		if (kernel.length != 5) {
			throw new RuntimeException("Incorrect call to convolve5");
		}
		double[] result = new double[vectorLength];
		for (int n = 2; n < vectorLength - 2; n++) {
			result[n+2] = vector[n-2]*kernel[0] +
			vector[n-1]*kernel[1] +
			vector[n]*kernel[2] +
			vector[n+1]*kernel[3] +
			vector[n+2]*kernel[4];
		}
		if (boundaryConditions == MIRROR_BOUNDARY) {
			result[0] = vector[2]*kernel[0]+vector[1]*kernel[1]+vector[0]*kernel[2]+vector[1]*kernel[3]+vector[2]*kernel[4];
			result[1] = vector[3]*kernel[0]+vector[2]*kernel[1]+vector[1]*kernel[2]+vector[0]*kernel[3]+vector[1]*kernel[4];
			result[vectorLength-1] = vector[vectorLength-3]*kernel[0] + vector[vectorLength-2]*kernel[1]+vector[vectorLength-1]*kernel[2] +
					vector[vectorLength-2]*kernel[3]+vector[vectorLength-3]*kernel[4];
			result[vectorLength-2] = vector[vectorLength-4]*kernel[0] + vector[vectorLength-3]*kernel[1]+vector[vectorLength-2]*kernel[2] +
					vector[vectorLength-1]*kernel[3]+vector[vectorLength-2]*kernel[4];
		} else if (boundaryConditions == PERIODIC_BOUNDARY) {
			result[0] = vector[2]*kernel[0]+vector[1]*kernel[1]+vector[0]*kernel[2]+vector[vectorLength-1]*kernel[3]+vector[vectorLength-2]*kernel[4];
			result[1] = vector[3]*kernel[0]+vector[2]*kernel[1]+vector[1]*kernel[2]+vector[0]*kernel[3]+vector[vectorLength-1]*kernel[4];
			result[vectorLength-1] = vector[vectorLength-3]*kernel[0] + vector[vectorLength-2]*kernel[1]+vector[vectorLength-1]*kernel[2] +
					vector[0]*kernel[3]+vector[1]*kernel[4];
			result[vectorLength-2] = vector[vectorLength-4]*kernel[0] + vector[vectorLength-3]*kernel[1]+vector[vectorLength-2]*kernel[2] +
					vector[vectorLength-1]*kernel[3]+vector[0]*kernel[4];
		}
		return result;
	}

	public double[][] convolve5(double[][] image, double[][] kernel) {
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageArea = imageWidth*imageHeight;
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
		if (kernelHeight != 5 || kernelWidth != 5) {
			throw new RuntimeException("Incorrect Call To Convolve5");
		}
		double[][] result = new double[imageWidth][imageHeight];
		int x, y, p, q, pq, adjX, adjY;
		for (int xy = 0; xy < imageArea; xy++) {
			x = imageArea % imageWidth;
			y = xy / imageWidth;
			boundaryCheck:
			if (x < 2 || x >= imageWidth || y < 2 || y >= imageHeight) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pq = 0; pq < 25; pq++) {
					p = pq % 5;
					q = pq / 5;
					adjX = x + (p - 2);
					adjY = y + (q - 2);
					if (adjX < 0) {
						if (boundaryConditions == MIRROR_BOUNDARY) {
							adjX = Math.abs(adjX);
						} else if (boundaryConditions == PERIODIC_BOUNDARY) {
							adjX = imageWidth + adjX;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					} else if (adjX > imageWidth) {
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
					} else if (adjY > imageHeight) {
						if (boundaryConditions == MIRROR_BOUNDARY) {
							adjY = 2*imageHeight - adjY - 1;
						} else if (boundaryConditions == PERIODIC_BOUNDARY) {
							adjY = adjY - imageHeight;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y] += image[adjX][adjY]*kernel[p][q];
				} // for pa
			} else {
					result[x][y] = 
						image[x-2][y-2]*kernel[-2][-2] + 
						image[x-2][y-1]*kernel[-2][-1] + 
						image[x-2][y+0]*kernel[-2][0] + 
						image[x-2][y+1]*kernel[-2][1] + 
						image[x-2][y+2]*kernel[-2][2] + 
						image[x-1][y-2]*kernel[-1][-2] + 
						image[x-1][y-1]*kernel[-1][-1] + 
						image[x-1][y+0]*kernel[-1][0] + 
						image[x-1][y+1]*kernel[-1][1] + 
						image[x-1][y+2]*kernel[-1][2] + 
						image[x+0][y-2]*kernel[0][-2] + 
						image[x+0][y-1]*kernel[0][-1] + 
						image[x+0][y+0]*kernel[0][0] + 
						image[x+0][y+1]*kernel[0][1] + 
						image[x+0][y+2]*kernel[0][2] + 
						image[x+1][y-2]*kernel[1][-2] + 
						image[x+1][y-1]*kernel[1][-1] + 
						image[x+1][y+0]*kernel[1][0] + 
						image[x+1][y+1]*kernel[1][1] + 
						image[x+1][y+2]*kernel[1][2] + 
						image[x+2][y-2]*kernel[2][-2] + 
						image[x+2][y-1]*kernel[2][-1] + 
						image[x+2][y+0]*kernel[2][0] + 
						image[x+2][y+1]*kernel[2][1] + 
						image[x+2][y+2]*kernel[2][2];
			} // if boundary
		} // for xy
		return result;
	}
	
	public double[][][] convolve5(double[][][] volume, double[][][] kernel) {

		int volumeWidth = volume[0].length;
		int volumeHeight = volume.length;
		int volumeDepth = volume[0][0].length;
		int totalArea = volumeWidth*volumeHeight;
		int totalVolume = totalArea*volumeDepth;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		int resultWidth = volumeWidth;
		int resultHeight = volumeHeight;
		int resultDepth = volumeDepth;
		
		if (kernelHeight != 5 || kernelWidth != 5 || kernelDepth != 5) {
			throw new RuntimeException("Incorrect Call To Convolve5");
		}

		double[][][] result = new double[resultWidth][resultHeight][resultDepth];
		int x, y, z, p, q, r, pqr, adjX, adjY, adjZ;
		for (int xyz = 0; xyz < totalVolume; xyz++) {
			z = xyz / totalArea;
			y = (xyz - z*totalArea) / volumeWidth;
			x = xyz % volumeWidth;
			boundaryCheck:
			if ( x < 2 || x >= volumeWidth - 2 || y < 2 || y >= volumeHeight - 2 || 
					z < 2 || z >= volumeDepth - 2) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pqr = 0; pqr < 125; pqr++) {
					r = pqr / 25;
					q = (pqr - r*25) / 5;
					p = pqr % 5;
					adjX = x + (p - 1);
					adjY = y + (q - 1);
					adjZ = z + (r - 1);
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
							adjZ = adjZ - volumeDepth;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y][z] += volume[adjX][adjY][adjZ]*kernel[p][q][r];
				} // for pqr
			} else { // if not boundary
				result[x][y][z] = 
						volume[x-2][y-2][z-2]*kernel[0][0][0] + 
						volume[x-2][y-2][z-1]*kernel[0][0][1] + 
						volume[x-2][y-2][z+0]*kernel[0][0][2] + 
						volume[x-2][y-2][z+1]*kernel[0][0][3] + 
						volume[x-2][y-2][z+2]*kernel[0][0][4] + 
						volume[x-2][y-1][z-2]*kernel[0][1][0] + 
						volume[x-2][y-1][z-1]*kernel[0][1][1] + 
						volume[x-2][y-1][z+0]*kernel[0][1][2] + 
						volume[x-2][y-1][z+1]*kernel[0][1][3] + 
						volume[x-2][y-1][z+2]*kernel[0][1][4] + 
						volume[x-2][y+0][z-2]*kernel[0][2][0] + 
						volume[x-2][y+0][z-1]*kernel[0][2][1] + 
						volume[x-2][y+0][z+0]*kernel[0][2][2] + 
						volume[x-2][y+0][z+1]*kernel[0][2][3] + 
						volume[x-2][y+0][z+2]*kernel[0][2][4] + 
						volume[x-2][y+1][z-2]*kernel[0][3][0] + 
						volume[x-2][y+1][z-1]*kernel[0][3][1] + 
						volume[x-2][y+1][z+0]*kernel[0][3][2] + 
						volume[x-2][y+1][z+1]*kernel[0][3][3] + 
						volume[x-2][y+1][z+2]*kernel[0][3][4] + 
						volume[x-2][y+2][z-2]*kernel[0][4][0] + 
						volume[x-2][y+2][z-1]*kernel[0][4][1] + 
						volume[x-2][y+2][z+0]*kernel[0][4][2] + 
						volume[x-2][y+2][z+1]*kernel[0][4][3] + 
						volume[x-2][y+2][z+2]*kernel[0][4][4] + 
						volume[x-1][y-2][z-2]*kernel[1][0][0] + 
						volume[x-1][y-2][z-1]*kernel[1][0][1] + 
						volume[x-1][y-2][z+0]*kernel[1][0][2] + 
						volume[x-1][y-2][z+1]*kernel[1][0][3] + 
						volume[x-1][y-2][z+2]*kernel[1][0][4] + 
						volume[x-1][y-1][z-2]*kernel[1][1][0] + 
						volume[x-1][y-1][z-1]*kernel[1][1][1] + 
						volume[x-1][y-1][z+0]*kernel[1][1][2] + 
						volume[x-1][y-1][z+1]*kernel[1][1][3] + 
						volume[x-1][y-1][z+2]*kernel[1][1][4] + 
						volume[x-1][y+0][z-2]*kernel[1][2][0] + 
						volume[x-1][y+0][z-1]*kernel[1][2][1] + 
						volume[x-1][y+0][z+0]*kernel[1][2][2] + 
						volume[x-1][y+0][z+1]*kernel[1][2][3] + 
						volume[x-1][y+0][z+2]*kernel[1][2][4] + 
						volume[x-1][y+1][z-2]*kernel[1][3][0] + 
						volume[x-1][y+1][z-1]*kernel[1][3][1] + 
						volume[x-1][y+1][z+0]*kernel[1][3][2] + 
						volume[x-1][y+1][z+1]*kernel[1][3][3] + 
						volume[x-1][y+1][z+2]*kernel[1][3][4] + 
						volume[x-1][y+2][z-2]*kernel[1][4][0] + 
						volume[x-1][y+2][z-1]*kernel[1][4][1] + 
						volume[x-1][y+2][z+0]*kernel[1][4][2] + 
						volume[x-1][y+2][z+1]*kernel[1][4][3] + 
						volume[x-1][y+2][z+2]*kernel[1][4][4] + 
						volume[x+0][y-2][z-2]*kernel[2][0][0] + 
						volume[x+0][y-2][z-1]*kernel[2][0][1] + 
						volume[x+0][y-2][z+0]*kernel[2][0][2] + 
						volume[x+0][y-2][z+1]*kernel[2][0][3] + 
						volume[x+0][y-2][z+2]*kernel[2][0][4] + 
						volume[x+0][y-1][z-2]*kernel[2][1][0] + 
						volume[x+0][y-1][z-1]*kernel[2][1][1] + 
						volume[x+0][y-1][z+0]*kernel[2][1][2] + 
						volume[x+0][y-1][z+1]*kernel[2][1][3] + 
						volume[x+0][y-1][z+2]*kernel[2][1][4] + 
						volume[x+0][y+0][z-2]*kernel[2][2][0] + 
						volume[x+0][y+0][z-1]*kernel[2][2][1] + 
						volume[x+0][y+0][z+0]*kernel[2][2][2] + 
						volume[x+0][y+0][z+1]*kernel[2][2][3] + 
						volume[x+0][y+0][z+2]*kernel[2][2][4] + 
						volume[x+0][y+1][z-2]*kernel[2][3][0] + 
						volume[x+0][y+1][z-1]*kernel[2][3][1] + 
						volume[x+0][y+1][z+0]*kernel[2][3][2] + 
						volume[x+0][y+1][z+1]*kernel[2][3][3] + 
						volume[x+0][y+1][z+2]*kernel[2][3][4] + 
						volume[x+0][y+2][z-2]*kernel[2][4][0] + 
						volume[x+0][y+2][z-1]*kernel[2][4][1] + 
						volume[x+0][y+2][z+0]*kernel[2][4][2] + 
						volume[x+0][y+2][z+1]*kernel[2][4][3] + 
						volume[x+0][y+2][z+2]*kernel[2][4][4] + 
						volume[x+1][y-2][z-2]*kernel[3][0][0] + 
						volume[x+1][y-2][z-1]*kernel[3][0][1] + 
						volume[x+1][y-2][z+0]*kernel[3][0][2] + 
						volume[x+1][y-2][z+1]*kernel[3][0][3] + 
						volume[x+1][y-2][z+2]*kernel[3][0][4] + 
						volume[x+1][y-1][z-2]*kernel[3][1][0] + 
						volume[x+1][y-1][z-1]*kernel[3][1][1] + 
						volume[x+1][y-1][z+0]*kernel[3][1][2] + 
						volume[x+1][y-1][z+1]*kernel[3][1][3] + 
						volume[x+1][y-1][z+2]*kernel[3][1][4] + 
						volume[x+1][y+0][z-2]*kernel[3][2][0] + 
						volume[x+1][y+0][z-1]*kernel[3][2][1] + 
						volume[x+1][y+0][z+0]*kernel[3][2][2] + 
						volume[x+1][y+0][z+1]*kernel[3][2][3] + 
						volume[x+1][y+0][z+2]*kernel[3][2][4] + 
						volume[x+1][y+1][z-2]*kernel[3][3][0] + 
						volume[x+1][y+1][z-1]*kernel[3][3][1] + 
						volume[x+1][y+1][z+0]*kernel[3][3][2] + 
						volume[x+1][y+1][z+1]*kernel[3][3][3] + 
						volume[x+1][y+1][z+2]*kernel[3][3][4] + 
						volume[x+1][y+2][z-2]*kernel[3][4][0] + 
						volume[x+1][y+2][z-1]*kernel[3][4][1] + 
						volume[x+1][y+2][z+0]*kernel[3][4][2] + 
						volume[x+1][y+2][z+1]*kernel[3][4][3] + 
						volume[x+1][y+2][z+2]*kernel[3][4][4] + 
						volume[x+2][y-2][z-2]*kernel[4][0][0] + 
						volume[x+2][y-2][z-1]*kernel[4][0][1] + 
						volume[x+2][y-2][z+0]*kernel[4][0][2] + 
						volume[x+2][y-2][z+1]*kernel[4][0][3] + 
						volume[x+2][y-2][z+2]*kernel[4][0][4] + 
						volume[x+2][y-1][z-2]*kernel[4][1][0] + 
						volume[x+2][y-1][z-1]*kernel[4][1][1] + 
						volume[x+2][y-1][z+0]*kernel[4][1][2] + 
						volume[x+2][y-1][z+1]*kernel[4][1][3] + 
						volume[x+2][y-1][z+2]*kernel[4][1][4] + 
						volume[x+2][y+0][z-2]*kernel[4][2][0] + 
						volume[x+2][y+0][z-1]*kernel[4][2][1] + 
						volume[x+2][y+0][z+0]*kernel[4][2][2] + 
						volume[x+2][y+0][z+1]*kernel[4][2][3] + 
						volume[x+2][y+0][z+2]*kernel[4][2][4] + 
						volume[x+2][y+1][z-2]*kernel[4][3][0] + 
						volume[x+2][y+1][z-1]*kernel[4][3][1] + 
						volume[x+2][y+1][z+0]*kernel[4][3][2] + 
						volume[x+2][y+1][z+1]*kernel[4][3][3] + 
						volume[x+2][y+1][z+2]*kernel[4][3][4] + 
						volume[x+2][y+2][z-2]*kernel[4][4][0] + 
						volume[x+2][y+2][z-1]*kernel[4][4][1] + 
						volume[x+2][y+2][z+0]*kernel[4][4][2] + 
						volume[x+2][y+2][z+1]*kernel[4][4][3] + 
						volume[x+2][y+2][z+2]*kernel[4][4][4];
			} // if not boundary
		} // for xyz
		return result;
		
	}

	public Complex[] convolve3(Complex[] vector, Complex[] kernel) {
		int vectorLength = vector.length;
		if (kernel.length != 3) {
			throw new RuntimeException("Incorrect call to convolve3");
		}
		Complex[] result = new Complex[vectorLength];
		for (int n = 1; n < vectorLength - 1; n++) {
			result[n+1]
			.add(vector[n-1].multiply(kernel[0]))
			.add(vector[n].multiply(kernel[1]))
			.add(vector[n+1].multiply(kernel[2]));
		}
		if (boundaryConditions == MIRROR_BOUNDARY) {
			result[0]
					.add(vector[1].multiply(kernel[0]))
					.add(vector[0].multiply(kernel[1]))
					.add(vector[1].multiply(kernel[2]));
			result[vectorLength-1]
					.add(vector[vectorLength-2].multiply(kernel[0]))
					.add(vector[vectorLength-1].multiply(kernel[1]))
					.add(vector[vectorLength-2].multiply(kernel[2]));
		} else if (boundaryConditions == PERIODIC_BOUNDARY) {
			result[0]
					.add(vector[vectorLength-1].multiply(kernel[0]))
					.add(vector[0].multiply(kernel[1]))
					.add(vector[1].multiply(kernel[2]));
			result[vectorLength-1]
					.add(vector[vectorLength-2].multiply(kernel[0]))
					.add(vector[vectorLength-1].multiply(kernel[1]))
					.add(vector[0].multiply(kernel[2]));
		}
		return result;
	}
	
	public Complex[][] convolve3(Complex[][] image, Complex[][] kernel) {
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageArea = imageWidth*imageHeight;
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
		if (kernelHeight != 3 || kernelWidth != 3) {
			throw new RuntimeException("Incorrect Call To Convolve3");
		}
		Complex[][] result = new Complex[imageWidth][imageHeight];
		int x, y, p, q, pq, adjX, adjY;
		for (int xy = 0; xy < imageArea; xy++) {
			x = imageArea % imageWidth;
			y = xy / imageWidth;
			boundaryCheck:
			if (x < 1 || x >= imageWidth - 1 || y < 1 || y >= imageHeight - 1) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pq = 0; pq < 9; pq++) {
					p = pq % 3;
					q = pq / 3;
					adjX = x + (p - 2);
					adjY = y + (q - 2);
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
					result[x][y].add(image[adjX][adjY].multiply(kernel[p][q]));
				} // for pq
			} else {
				result[x][y]
					.add(image[x-1][y-1].multiply(kernel[-1][-1]))
					.add(image[x-1][y+0].multiply(kernel[-1][0]))
					.add(image[x-1][y+1].multiply(kernel[-1][1]))
					.add(image[x+0][y-1].multiply(kernel[0][-1]))
					.add(image[x+0][y+0].multiply(kernel[0][0]))
					.add(image[x+0][y+1].multiply(kernel[0][1]))
					.add(image[x+1][y-1].multiply(kernel[1][-1]))
					.add(image[x+1][y+0].multiply(kernel[1][0]))
					.add(image[x+1][y+1].multiply(kernel[1][1]));
			} // if boundary
		} // for xy
		return result;
	}

	public Complex[][][] convolve3(Complex[][][] volume, Complex[][][] kernel) {
		
		int volumeWidth = volume[0].length;
		int volumeHeight = volume.length;
		int volumeDepth = volume[0][0].length;
		int totalArea = volumeWidth*volumeHeight;
		int totalVolume = totalArea*volumeDepth;
		int kernelWidth = kernel[0].length;
		int kernelHeight = kernel.length;
		int kernelDepth = kernel[0][0].length;
		int kernelArea = kernelWidth*kernelHeight;
		if (kernelHeight != 3 || kernelWidth != 3 || kernelDepth != 3) {
			throw new RuntimeException("Incorrect Call To Convolve3");
		}
		Complex[][][] result = new Complex[volumeWidth][volumeHeight][volumeDepth];
		int x, y, z, p, q, r, pqr, adjX, adjY, adjZ;
		for (int xyz = 0; xyz < totalVolume; xyz++) {
			z = xyz / totalArea;
			y = (xyz - z*totalArea) / volumeWidth;
			x = xyz % volumeWidth;
			boundaryCheck:
			if ( x < 1 || x >= volumeWidth-1 || y < 1 || y >= volumeHeight-1 || 
					z < 1 || z >= volumeHeight-1 ) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pqr = 0; pqr < 27; pqr++) {
					r = pqr / 9;
					q = (pqr - r*9) / 3;
					p = pqr % 3;
					adjX = x + (p - 1);
					adjY = y + (q - 1);
					adjZ = z + (r - 1);
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
							adjZ = adjZ - volumeDepth;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y][z].add(volume[adjX][adjY][adjZ].multiply(kernel[p][q][r]));
				} // for pqr
			} else { // if not boundary
					result[x][y][z]
						.add(volume[x-1][y-1][z-1].multiply(kernel[-1][-1][-1]))
						.add(volume[x-1][y-1][z+0].multiply(kernel[-1][-1][0]))
						.add(volume[x-1][y-1][z+1].multiply(kernel[-1][-1][1]))
						.add(volume[x-1][y+0][z-1].multiply(kernel[-1][0][-1]))
						.add(volume[x-1][y+0][z+0].multiply(kernel[-1][0][0]))
						.add(volume[x-1][y+0][z+1].multiply(kernel[-1][0][1]))
						.add(volume[x-1][y+1][z-1].multiply(kernel[-1][1][-1]))
						.add(volume[x-1][y+1][z+0].multiply(kernel[-1][1][0]))
						.add(volume[x-1][y+1][z+1].multiply(kernel[-1][1][1]))
						.add(volume[x+0][y-1][z-1].multiply(kernel[0][-1][-1]))
						.add(volume[x+0][y-1][z+0].multiply(kernel[0][-1][0]))
						.add(volume[x+0][y-1][z+1].multiply(kernel[0][-1][1]))
						.add(volume[x+0][y+0][z-1].multiply(kernel[0][0][-1]))
						.add(volume[x+0][y+0][z+0].multiply(kernel[0][0][0]))
						.add(volume[x+0][y+0][z+1].multiply(kernel[0][0][1]))
						.add(volume[x+0][y+1][z-1].multiply(kernel[0][1][-1]))
						.add(volume[x+0][y+1][z+0].multiply(kernel[0][1][0]))
						.add(volume[x+0][y+1][z+1].multiply(kernel[0][1][1]))
						.add(volume[x+1][y-1][z-1].multiply(kernel[1][-1][-1]))
						.add(volume[x+1][y-1][z+0].multiply(kernel[1][-1][0]))
						.add(volume[x+1][y-1][z+1].multiply(kernel[1][-1][1]))
						.add(volume[x+1][y+0][z-1].multiply(kernel[1][0][-1]))
						.add(volume[x+1][y+0][z+0].multiply(kernel[1][0][0]))
						.add(volume[x+1][y+0][z+1].multiply(kernel[1][0][1]))
						.add(volume[x+1][y+1][z-1].multiply(kernel[1][1][-1]))
						.add(volume[x+1][y+1][z+0].multiply(kernel[1][1][0]))
						.add(volume[x+1][y+1][z+1].multiply(kernel[1][1][1]));
			} // if not boundary
		} // for xyz
		return result;
	}
	
	public Complex[] convolve5(Complex[] vector, Complex[] kernel) {
		int vectorLength = vector.length;
		if (kernel.length != 5) {
			throw new RuntimeException("Incorrect call to convolve5");
		}
		Complex[] result = new Complex[vectorLength];
		for (int n = 2; n < vectorLength - 2; n++) {
			result[n+2]
					.add(vector[n-2].multiply(kernel[0]))
					.add(vector[n-1].multiply(kernel[1]))
					.add(vector[n].multiply(kernel[2]))
					.add(vector[n+1].multiply(kernel[3]))
					.add(vector[n+2].multiply(kernel[4]));
		}
		if (boundaryConditions == MIRROR_BOUNDARY) {
			result[0]
					.add(vector[2].multiply(kernel[0]))
					.add(vector[1].multiply(kernel[1]))
					.add(vector[0].multiply(kernel[2]))
					.add(vector[1].multiply(kernel[3]))
					.add(vector[2].multiply(kernel[4]));
			result[1]
					.add(vector[3].multiply(kernel[0]))
					.add(vector[2].multiply(kernel[1]))
					.add(vector[1].multiply(kernel[2]))
					.add(vector[0].multiply(kernel[3]))
					.add(vector[1].multiply(kernel[4]));
			result[vectorLength-1]
					.add(vector[vectorLength-3].multiply(kernel[0]))
					.add(vector[vectorLength-2].multiply(kernel[1]))
					.add(vector[vectorLength-1].multiply(kernel[2]))
					.add(vector[vectorLength-2].multiply(kernel[3]))
					.add(vector[vectorLength-3].multiply(kernel[4]));
			result[vectorLength-2]
					.add(vector[vectorLength-4].multiply(kernel[0]))
					.add(vector[vectorLength-3].multiply(kernel[1]))
					.add(vector[vectorLength-2].multiply(kernel[2]))
					.add(vector[vectorLength-1].multiply(kernel[3]))
					.add(vector[vectorLength-2].multiply(kernel[4]));
		} else if (boundaryConditions == PERIODIC_BOUNDARY) {
			result[0]
					.add(vector[2].multiply(kernel[0]))
					.add(vector[1].multiply(kernel[1]))
					.add(vector[0].multiply(kernel[2]))
					.add(vector[vectorLength-1].multiply(kernel[3]))
					.add(vector[vectorLength-2].multiply(kernel[4]));
			result[1]
					.add(vector[3].multiply(kernel[0]))
					.add(vector[2].multiply(kernel[1]))
					.add(vector[1].multiply(kernel[2]))
					.add(vector[0].multiply(kernel[3]))
					.add(vector[vectorLength-1].multiply(kernel[4]));
			result[vectorLength-1]
					.add(vector[vectorLength-3].multiply(kernel[0]))
					.add( vector[vectorLength-2].multiply(kernel[1]))
					.add(vector[vectorLength-1].multiply(kernel[2]))
					.add(vector[0].multiply(kernel[3]))
					.add(vector[1].multiply(kernel[4]));
			result[vectorLength-2]
					.add(vector[vectorLength-4].multiply(kernel[0]))
					.add( vector[vectorLength-3].multiply(kernel[1]))
					.add(vector[vectorLength-2].multiply(kernel[2]))
					.add(vector[vectorLength-1].multiply(kernel[3]))
					.add(vector[0].multiply(kernel[4]));
		}
		return result;
	}

	public Complex[][] convolve5(Complex[][] image, Complex[][] kernel) {
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageArea = imageWidth*imageHeight;
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
		if (kernelHeight != 5 || kernelWidth != 5) {
			throw new RuntimeException("Incorrect Call To Convolve5");
		}
		Complex[][] result = new Complex[imageWidth][imageHeight];
		int x, y, p, q, pq, adjX, adjY;
		for (int xy = 0; xy < imageArea; xy++) {
			x = imageArea % imageWidth;
			y = xy / imageWidth;
			boundaryCheck:
			if (x < 2 || x >= imageWidth || y < 2 || y >= imageHeight) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pq = 0; pq < 25; pq++) {
					p = pq % 5;
					q = pq / 5;
					adjX = x + (p - 2);
					adjY = y + (q - 2);
					if (adjX < 0) {
						if (boundaryConditions == MIRROR_BOUNDARY) {
							adjX = Math.abs(adjX);
						} else if (boundaryConditions == PERIODIC_BOUNDARY) {
							adjX = imageWidth + adjX;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					} else if (adjX > imageWidth) {
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
					} else if (adjY > imageHeight) {
						if (boundaryConditions == MIRROR_BOUNDARY) {
							adjY = 2*imageHeight - adjY - 1;
						} else if (boundaryConditions == PERIODIC_BOUNDARY) {
							adjY = adjY - imageHeight;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y].add(image[adjX][adjY].multiply(kernel[p][q]));
				} // for pa
			} else {
					result[x][y]
						.add(image[x-2][y-2].multiply(kernel[-2][-2]))
						.add(image[x-2][y-1].multiply(kernel[-2][-1]))
						.add(image[x-2][y+0].multiply(kernel[-2][0]))
						.add(image[x-2][y+1].multiply(kernel[-2][1]))
						.add(image[x-2][y+2].multiply(kernel[-2][2]))
						.add(image[x-1][y-2].multiply(kernel[-1][-2]))
						.add(image[x-1][y-1].multiply(kernel[-1][-1]))
						.add(image[x-1][y+0].multiply(kernel[-1][0]))
						.add(image[x-1][y+1].multiply(kernel[-1][1]))
						.add(image[x-1][y+2].multiply(kernel[-1][2]))
						.add(image[x+0][y-2].multiply(kernel[0][-2]))
						.add(image[x+0][y-1].multiply(kernel[0][-1]))
						.add(image[x+0][y+0].multiply(kernel[0][0]))
						.add(image[x+0][y+1].multiply(kernel[0][1]))
						.add(image[x+0][y+2].multiply(kernel[0][2]))
						.add(image[x+1][y-2].multiply(kernel[1][-2]))
						.add(image[x+1][y-1].multiply(kernel[1][-1]))
						.add(image[x+1][y+0].multiply(kernel[1][0]))
						.add(image[x+1][y+1].multiply(kernel[1][1]))
						.add(image[x+1][y+2].multiply(kernel[1][2]))
						.add(image[x+2][y-2].multiply(kernel[2][-2]))
						.add(image[x+2][y-1].multiply(kernel[2][-1]))
						.add(image[x+2][y+0].multiply(kernel[2][0]))
						.add(image[x+2][y+1].multiply(kernel[2][1]))
						.add(image[x+2][y+2].multiply(kernel[2][2]));
			} // if boundary
		} // for xy
		return result;
	}
	
	public Complex[][][] convolve5(Complex[][][] volume, Complex[][][] kernel) {

		int volumeWidth = volume[0].length;
		int volumeHeight = volume.length;
		int volumeDepth = volume[0][0].length;
		int totalArea = volumeWidth*volumeHeight;
		int totalVolume = totalArea*volumeDepth;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		int resultWidth = volumeWidth;
		int resultHeight = volumeHeight;
		int resultDepth = volumeDepth;
		
		if (kernelHeight != 5 || kernelWidth != 5 || kernelDepth != 5) {
			throw new RuntimeException("Incorrect Call To Convolve5");
		}

		Complex[][][] result = new Complex[resultWidth][resultHeight][resultDepth];
		int x, y, z, p, q, r, pqr, adjX, adjY, adjZ;
		for (int xyz = 0; xyz < totalVolume; xyz++) {
			z = xyz / totalArea;
			y = (xyz - z*totalArea) / volumeWidth;
			x = xyz % volumeWidth;
			boundaryCheck:
			if ( x < 2 || x >= volumeWidth - 2 || y < 2 || y >= volumeHeight - 2 || 
					z < 2 || z >= volumeDepth - 2) {
				if (boundaryConditions == ZERO_BOUNDARY) break boundaryCheck;
				for (pqr = 0; pqr < 125; pqr++) {
					r = pqr / 25;
					q = (pqr - r*25) / 5;
					p = pqr % 5;
					adjX = x + (p - 1);
					adjY = y + (q - 1);
					adjZ = z + (r - 1);
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
							adjZ = adjZ - volumeDepth;
						} else throw new RuntimeException("Exception in FD boundaries");// BCs
					}
					result[x][y][z].add(volume[adjX][adjY][adjZ].multiply(kernel[p][q][r]));
				} // for pqr
			} else { // if not boundary
				result[x][y][z]
					.add(volume[x-2][y-2][z-2].multiply(kernel[-2][-2][-2]))
					.add(volume[x-2][y-2][z-1].multiply(kernel[-2][-2][-1]))
					.add(volume[x-2][y-2][z+0].multiply(kernel[-2][-2][0]))
					.add(volume[x-2][y-2][z+1].multiply(kernel[-2][-2][1]))
					.add(volume[x-2][y-2][z+2].multiply(kernel[-2][-2][2]))
					.add(volume[x-2][y-1][z-2].multiply(kernel[-2][-1][-2]))
					.add(volume[x-2][y-1][z-1].multiply(kernel[-2][-1][-1]))
					.add(volume[x-2][y-1][z+0].multiply(kernel[-2][-1][0]))
					.add(volume[x-2][y-1][z+1].multiply(kernel[-2][-1][1]))
					.add(volume[x-2][y-1][z+2].multiply(kernel[-2][-1][2]))
					.add(volume[x-2][y+0][z-2].multiply(kernel[-2][0][-2]))
					.add(volume[x-2][y+0][z-1].multiply(kernel[-2][0][-1]))
					.add(volume[x-2][y+0][z+0].multiply(kernel[-2][0][0]))
					.add(volume[x-2][y+0][z+1].multiply(kernel[-2][0][1]))
					.add(volume[x-2][y+0][z+2].multiply(kernel[-2][0][2]))
					.add(volume[x-2][y+1][z-2].multiply(kernel[-2][1][-2]))
					.add(volume[x-2][y+1][z-1].multiply(kernel[-2][1][-1]))
					.add(volume[x-2][y+1][z+0].multiply(kernel[-2][1][0]))
					.add(volume[x-2][y+1][z+1].multiply(kernel[-2][1][1]))
					.add(volume[x-2][y+1][z+2].multiply(kernel[-2][1][2]))
					.add(volume[x-2][y+2][z-2].multiply(kernel[-2][2][-2]))
					.add(volume[x-2][y+2][z-1].multiply(kernel[-2][2][-1]))
					.add(volume[x-2][y+2][z+0].multiply(kernel[-2][2][0]))
					.add(volume[x-2][y+2][z+1].multiply(kernel[-2][2][1]))
					.add(volume[x-2][y+2][z+2].multiply(kernel[-2][2][2]))
					.add(volume[x-1][y-2][z-2].multiply(kernel[-1][-2][-2]))
					.add(volume[x-1][y-2][z-1].multiply(kernel[-1][-2][-1]))
					.add(volume[x-1][y-2][z+0].multiply(kernel[-1][-2][0]))
					.add(volume[x-1][y-2][z+1].multiply(kernel[-1][-2][1]))
					.add(volume[x-1][y-2][z+2].multiply(kernel[-1][-2][2]))
					.add(volume[x-1][y-1][z-2].multiply(kernel[-1][-1][-2]))
					.add(volume[x-1][y-1][z-1].multiply(kernel[-1][-1][-1]))
					.add(volume[x-1][y-1][z+0].multiply(kernel[-1][-1][0]))
					.add(volume[x-1][y-1][z+1].multiply(kernel[-1][-1][1]))
					.add(volume[x-1][y-1][z+2].multiply(kernel[-1][-1][2]))
					.add(volume[x-1][y+0][z-2].multiply(kernel[-1][0][-2]))
					.add(volume[x-1][y+0][z-1].multiply(kernel[-1][0][-1]))
					.add(volume[x-1][y+0][z+0].multiply(kernel[-1][0][0]))
					.add(volume[x-1][y+0][z+1].multiply(kernel[-1][0][1]))
					.add(volume[x-1][y+0][z+2].multiply(kernel[-1][0][2]))
					.add(volume[x-1][y+1][z-2].multiply(kernel[-1][1][-2]))
					.add(volume[x-1][y+1][z-1].multiply(kernel[-1][1][-1]))
					.add(volume[x-1][y+1][z+0].multiply(kernel[-1][1][0]))
					.add(volume[x-1][y+1][z+1].multiply(kernel[-1][1][1]))
					.add(volume[x-1][y+1][z+2].multiply(kernel[-1][1][2]))
					.add(volume[x-1][y+2][z-2].multiply(kernel[-1][2][-2]))
					.add(volume[x-1][y+2][z-1].multiply(kernel[-1][2][-1]))
					.add(volume[x-1][y+2][z+0].multiply(kernel[-1][2][0]))
					.add(volume[x-1][y+2][z+1].multiply(kernel[-1][2][1]))
					.add(volume[x-1][y+2][z+2].multiply(kernel[-1][2][2]))
					.add(volume[x+0][y-2][z-2].multiply(kernel[0][-2][-2]))
					.add(volume[x+0][y-2][z-1].multiply(kernel[0][-2][-1]))
					.add(volume[x+0][y-2][z+0].multiply(kernel[0][-2][0]))
					.add(volume[x+0][y-2][z+1].multiply(kernel[0][-2][1]))
					.add(volume[x+0][y-2][z+2].multiply(kernel[0][-2][2]))
					.add(volume[x+0][y-1][z-2].multiply(kernel[0][-1][-2]))
					.add(volume[x+0][y-1][z-1].multiply(kernel[0][-1][-1]))
					.add(volume[x+0][y-1][z+0].multiply(kernel[0][-1][0]))
					.add(volume[x+0][y-1][z+1].multiply(kernel[0][-1][1]))
					.add(volume[x+0][y-1][z+2].multiply(kernel[0][-1][2]))
					.add(volume[x+0][y+0][z-2].multiply(kernel[0][0][-2]))
					.add(volume[x+0][y+0][z-1].multiply(kernel[0][0][-1]))
					.add(volume[x+0][y+0][z+0].multiply(kernel[0][0][0]))
					.add(volume[x+0][y+0][z+1].multiply(kernel[0][0][1]))
					.add(volume[x+0][y+0][z+2].multiply(kernel[0][0][2]))
					.add(volume[x+0][y+1][z-2].multiply(kernel[0][1][-2]))
					.add(volume[x+0][y+1][z-1].multiply(kernel[0][1][-1]))
					.add(volume[x+0][y+1][z+0].multiply(kernel[0][1][0]))
					.add(volume[x+0][y+1][z+1].multiply(kernel[0][1][1]))
					.add(volume[x+0][y+1][z+2].multiply(kernel[0][1][2]))
					.add(volume[x+0][y+2][z-2].multiply(kernel[0][2][-2]))
					.add(volume[x+0][y+2][z-1].multiply(kernel[0][2][-1]))
					.add(volume[x+0][y+2][z+0].multiply(kernel[0][2][0]))
					.add(volume[x+0][y+2][z+1].multiply(kernel[0][2][1]))
					.add(volume[x+0][y+2][z+2].multiply(kernel[0][2][2]))
					.add(volume[x+1][y-2][z-2].multiply(kernel[1][-2][-2]))
					.add(volume[x+1][y-2][z-1].multiply(kernel[1][-2][-1]))
					.add(volume[x+1][y-2][z+0].multiply(kernel[1][-2][0]))
					.add(volume[x+1][y-2][z+1].multiply(kernel[1][-2][1]))
					.add(volume[x+1][y-2][z+2].multiply(kernel[1][-2][2]))
					.add(volume[x+1][y-1][z-2].multiply(kernel[1][-1][-2]))
					.add(volume[x+1][y-1][z-1].multiply(kernel[1][-1][-1]))
					.add(volume[x+1][y-1][z+0].multiply(kernel[1][-1][0]))
					.add(volume[x+1][y-1][z+1].multiply(kernel[1][-1][1]))
					.add(volume[x+1][y-1][z+2].multiply(kernel[1][-1][2]))
					.add(volume[x+1][y+0][z-2].multiply(kernel[1][0][-2]))
					.add(volume[x+1][y+0][z-1].multiply(kernel[1][0][-1]))
					.add(volume[x+1][y+0][z+0].multiply(kernel[1][0][0]))
					.add(volume[x+1][y+0][z+1].multiply(kernel[1][0][1]))
					.add(volume[x+1][y+0][z+2].multiply(kernel[1][0][2]))
					.add(volume[x+1][y+1][z-2].multiply(kernel[1][1][-2]))
					.add(volume[x+1][y+1][z-1].multiply(kernel[1][1][-1]))
					.add(volume[x+1][y+1][z+0].multiply(kernel[1][1][0]))
					.add(volume[x+1][y+1][z+1].multiply(kernel[1][1][1]))
					.add(volume[x+1][y+1][z+2].multiply(kernel[1][1][2]))
					.add(volume[x+1][y+2][z-2].multiply(kernel[1][2][-2]))
					.add(volume[x+1][y+2][z-1].multiply(kernel[1][2][-1]))
					.add(volume[x+1][y+2][z+0].multiply(kernel[1][2][0]))
					.add(volume[x+1][y+2][z+1].multiply(kernel[1][2][1]))
					.add(volume[x+1][y+2][z+2].multiply(kernel[1][2][2]))
					.add(volume[x+2][y-2][z-2].multiply(kernel[2][-2][-2]))
					.add(volume[x+2][y-2][z-1].multiply(kernel[2][-2][-1]))
					.add(volume[x+2][y-2][z+0].multiply(kernel[2][-2][0]))
					.add(volume[x+2][y-2][z+1].multiply(kernel[2][-2][1]))
					.add(volume[x+2][y-2][z+2].multiply(kernel[2][-2][2]))
					.add(volume[x+2][y-1][z-2].multiply(kernel[2][-1][-2]))
					.add(volume[x+2][y-1][z-1].multiply(kernel[2][-1][-1]))
					.add(volume[x+2][y-1][z+0].multiply(kernel[2][-1][0]))
					.add(volume[x+2][y-1][z+1].multiply(kernel[2][-1][1]))
					.add(volume[x+2][y-1][z+2].multiply(kernel[2][-1][2]))
					.add(volume[x+2][y+0][z-2].multiply(kernel[2][0][-2]))
					.add(volume[x+2][y+0][z-1].multiply(kernel[2][0][-1]))
					.add(volume[x+2][y+0][z+0].multiply(kernel[2][0][0]))
					.add(volume[x+2][y+0][z+1].multiply(kernel[2][0][1]))
					.add(volume[x+2][y+0][z+2].multiply(kernel[2][0][2]))
					.add(volume[x+2][y+1][z-2].multiply(kernel[2][1][-2]))
					.add(volume[x+2][y+1][z-1].multiply(kernel[2][1][-1]))
					.add(volume[x+2][y+1][z+0].multiply(kernel[2][1][0]))
					.add(volume[x+2][y+1][z+1].multiply(kernel[2][1][1]))
					.add(volume[x+2][y+1][z+2].multiply(kernel[2][1][2]))
					.add(volume[x+2][y+2][z-2].multiply(kernel[2][2][-2]))
					.add(volume[x+2][y+2][z-1].multiply(kernel[2][2][-1]))
					.add(volume[x+2][y+2][z+0].multiply(kernel[2][2][0]))
					.add(volume[x+2][y+2][z+1].multiply(kernel[2][2][1]))
					.add(volume[x+2][y+2][z+2].multiply(kernel[2][2][2]));

			} // if not boundary
		} // for xyz
		return result;
		
	}
	
	public double[] convolve10(double[] f, double[] g, double[] r) {
		final int fi = f.length;
		final int gi = g.length;
		final int ri = r.length;
		for (int x = 5; x <= ri-5; x++) {
			r[x] = 
				f[x-5]*g[9] + 
				f[x-4]*g[8] + 
				f[x-3]*g[7] + 
				f[x-2]*g[6] + 
				f[x-1]*g[5] + 
				f[x]*g[4] + 
				f[x+1]*g[3] + 
				f[x+2]*g[2] + 
				f[x+3]*g[1] + 
				f[x+4]*g[0];
		}
		r[0] = f[0]*g[4] + f[1]*g[3] + f[2]*g[2] + f[3]*g[1] + f[4]*g[0];
		r[1] = f[0]*g[5] + f[1]*g[4] + f[2]*g[3] + f[3]*g[2] + f[4]*g[1] + f[5]*g[0];
		r[2] = f[0]*g[6] + f[1]*g[5] + f[2]*g[4] + f[3]*g[3] + f[4]*g[2]  + f[5]*g[1] + f[6]*g[0];
		r[3] = f[0]*g[7] + f[1]*g[6] + f[2]*g[5] + f[3]*g[4] + f[4]*g[3]  + f[5]*g[2] + f[6]*g[1] + f[7]*g[0];
		r[4] = f[0]*g[8] + f[1]*g[7] + f[2]*g[6] + f[3]*g[5] + f[4]*g[4]  + f[5]*g[3] + f[6]*g[2] + f[7]*g[1] + f[8]*g[0];
		r[ri-1] = f[ri-1]*g[4] + f[ri-2]*g[5] + f[ri-3]*g[6] + f[ri-4]*g[7] + f[ri-5]*g[8] + f[ri-6]*g[9];
		r[ri-2] = f[ri-1]*g[3] + f[ri-2]*g[4] + f[ri-3]*g[5] + f[ri-4]*g[6] + f[ri-5]*g[7] + f[ri-6]*g[8] + f[ri-7]*g[9];
		r[ri-3] = f[ri-1]*g[2] + f[ri-2]*g[3] + f[ri-3]*g[4] + f[ri-4]*g[5] + f[ri-5]*g[6] + f[ri-6]*g[7] + f[ri-7]*g[8] + f[ri-8]*g[9];
		r[ri-4] = f[ri-1]*g[1] + f[ri-2]*g[2] + f[ri-3]*g[3] + f[ri-4]*g[4] + f[ri-5]*g[5] + f[ri-6]*g[6] + f[ri-7]*g[7] + f[ri-8]*g[8] + f[ri-9]*g[9];
		return r;
	}
	
	public double[][] convolve10T(double[][] f, double[] g, int dim) {
		if ( dim == 1) {
			f = JVCLUtils.shiftDim(f);
		}
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgi, 0, 0);
		double[][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgi, 0, 0);
		for (int x = 0; x < fi; x++) {
			Runnable d = new DTConvolver2d(fPad, g, r, x);
			threadPool.execute(d);
		}
		threadPool.shutdown();
        while (!threadPool.isTerminated()) {
        }
        if ( dim == 1) {
			f = JVCLUtils.shiftDim(r);
		}
        return r;
	}
	
	public double[][] convolve10(double[][] f, double[] g, int dim) {
		if ( dim == 1) {
			f = JVCLUtils.shiftDim(f);
		}
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[][] fPad = JVCLUtils.zeroPadBoundaries(f, 0, 0, hgi, hgie);
		double[][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj], 0, 0, hgi, hgie);
		for (int x = 0; x < fi; x++) {
			r[x] = convolve10(fPad[x], g, r[x]);
		}
		if ( dim == 1) {
			f = JVCLUtils.shiftDim(r);
		}
        return r;
	}
	
	public static void main(String[] args) {
		double[] g = new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
		double[][] f = new double[64][64];
		for (int n = 0; n < 64; n++) {
			f[n] = JVCLUtils.fillWithGradient(64);
		}
		FDCPUUnrolled u = new FDCPUUnrolled();
		long t1 = System.currentTimeMillis();
		//double[][] ft = u.convolve10T(f, g, 0);
		long t2 = System.currentTimeMillis();
		double[][] fnt = u.convolve10(f, g, 0);
		long t3 = System.currentTimeMillis();
		System.out.format("Threads %.3f No Threads %.3f %n", (t2-t1)/1000.0, (t3-t2)/1000.0);
		JVCLUtils.display2DArray(fnt, fnt.length, fnt[0].length, fnt.length, 0);
	}
	
	
	
}
