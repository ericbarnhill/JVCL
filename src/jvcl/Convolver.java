package jvcl;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;


/*
 * run part not needing attention to boundaries
 * handle boundaries with ensuing if statements
 * also adjust size of result at that time
 * 3 and 5 isotropic kernels are unrolled for speed
 * you can use the Unroller codegen class to unroll larger 
 * or custom size kernels
 * for your own fork if you wish
 * 
 * note to self: cropping for boundary conditions is measuring in terms of 
 * ORIG DIMENSION not RESULT DIMENSION
 * 
 * also, leave non-unrolled classes to take care of all boundaries
 */


public class Convolver {
	
	int boundaryConditions;
	
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	ConvolveNaive cn;
	ConvolveFourier cf;
	ConvolveJOCL cj;
	GPUFFT g;
	CPUFFT c;
	Stockham s;
	
	public Convolver(int boundaryConditions) {
		this.boundaryConditions = boundaryConditions;
		cn = new ConvolveNaive(boundaryConditions);
		cf = new ConvolveFourier();
		cj = new ConvolveJOCL();
		g = new GPUFFT(4096, 256, 256);
		c = new CPUFFT();
		s = new Stockham();
	}
	
	public Convolver() {
		this.boundaryConditions = MIRROR_BOUNDARY;
		cn = new ConvolveNaive(boundaryConditions);
		cf = new ConvolveFourier();
		cj = new ConvolveJOCL();
		g = new GPUFFT(4096, 256, 256);
		c = new CPUFFT();
		s = new Stockham();
	}
	
	public static void main(String[] args) {
		
        Path currentRelativePath = Paths.get("");
        String s = currentRelativePath.toAbsolutePath().toString();
        System.out.println(s);
		Random rand = new Random();
		Convolver c = new Convolver();
		float[] testvec1d = new float[128*128*4];
		for (int n = 0; n < 128*128*4; n++) {
			testvec1d[n] = (float)Math.cos(n*2*Math.PI/2048);
		}
		for (int n = 0; n < 24; n++) {
			//System.out.print(testvec1d[n]+" ");
		}
		System.out.println();
		double[] kernel1d = new double[]{1, -2, 1};
		long startTime = System.currentTimeMillis();
		for (int n = 0; n < 5; n++) {
			c.s.fft_simple(testvec1d, testvec1d);
		}
		long endTime = System.currentTimeMillis();
		double runTime = (endTime-startTime)/1000.0;
		startTime = System.currentTimeMillis();
		for (int n = 0; n < 5; n++) {
			c.c.fft_simple(testvec1d, testvec1d);
		}
		endTime = System.currentTimeMillis();
		double runTime2 = (endTime-startTime)/1000.0;
		System.out.format("cpu %.2f stockham %.2f", runTime2, runTime);
		if (true) return;
		int vectorWidth = 64;
		int vectorHeight = 64;
		int vectorDepth = 64;
		double[][][] vector = new double[vectorWidth][vectorHeight][vectorDepth];
		double[] kernelVals = new double[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 1, 0, 1, 0,
				0, 0, 0, 0, 1, 0, 0, 0, 0};
		int kDim = 15;
		double[][][] kernel = new double[kDim][kDim][kDim];
		for (int x = 0; x < kDim; x++) {
			for (int y = 0; y < kDim; y++) {
				for (int z = 0; z < kDim; z++) {
					kernel[x][y][z] = 1; //kernelVals[x + y*3 + z*9];
				}
			}
		}
		System.out.println("OPENCL");
		int trials = 15;
		double avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < vectorWidth; i++) {
				for (int j=0; j < vectorHeight; j++) {
					for (int k = 0; k < vectorDepth; k++) {
						vector[i][j][k] = i*i + j*j + k*k;
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result;
			try {
				result = c.cj.convolveFDJOCL(vector, kernel);
				for (int i = 2; i < 8; i++) {
					for (int j = 2; j < 8; j++) {
						for (int k = 2; k < 8; k++) {
						System.out.format("%.3f ",result[i][j][k]);
						}
					}
					System.out.format("%n");
				}
			} catch (Throwable t) {
				t.printStackTrace();
			}			
			if (true) return;
			long time2 = System.currentTimeMillis();
			runTime = (time2-time1)/1000.0;
			if (n >= 5) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
			for (int i = 0; i < vectorWidth; i++) {
				for (int j = 0; j < vectorHeight; j++) {
					for (int k = 0; k < vectorDepth; k++) {
					System.out.format("%.3f ",result[i][j][k]);
					}
				}
				System.out.format("%n");
			}
		}
		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-5));
		//--------------------------------
		System.out.println("NAIVE");
		avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < vectorWidth; i++) {
				for (int j=0; j < vectorHeight; j++) {
					for (int k = 0; k < vectorDepth; k++) {
						vector[i][j][k] = i*i + j*j + k*k;
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result;
			try {
				result = c.cn.convolveFD(vector, kernel, true);
				//c.convolveJOCL(vector[0], kernel[0]);
				/*
				for (int i = 0; i < vectorWidth; i++) {
					for (int j = 0; j < vectorHeight; j++) {
						for (int k = 0; k < vectorDepth; k++) {
						System.out.format("%.3f ",result[i][j][k]);
						}
					}
					System.out.format("%n");
				}
				*/
			} catch (Throwable t) {
				t.printStackTrace();
			}			
			long time2 = System.currentTimeMillis();
			runTime = (time2-time1)/1000.0;
			if (n >= 5) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
		}

		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-5));
		// --------------------------------------------------
		System.out.println("UNROLL");
		avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < vectorWidth; i++) {
				for (int j=0; j < vectorHeight; j++) {
					for (int k = 0; k < vectorDepth; k++) {
						vector[i][j][k] = i*i + j*j + k*k;
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result;
			try {
				result = c.cn.convolveFD(vector, kernel, false);
				//c.convolveJOCL(vector[0], kernel[0]);
				/*
				for (int i = 0; i < vectorWidth; i++) {
					for (int j = 0; j < vectorHeight; j++) {
						for (int k = 0; k < vectorDepth; k++) {
						System.out.format("%.3f ",result[i][j][k]);
						}
					}
					System.out.format("%n");
				}
				*/
			} catch (Throwable t) {
				t.printStackTrace();
			}			
			long time2 = System.currentTimeMillis();
			runTime = (time2-time1)/1000.0;
			if (n >= 5) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
		}
		
		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-5));
		//-------------------------------------
		System.out.println("FFT");
		avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < vectorWidth; i++) {
				for (int j=0; j < vectorHeight; j++) {
					for (int k = 0; k < vectorDepth; k++) {
						vector[i][j][k] = i*i + j*j + k*k;
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result;
			try {
				result = c.cf.convolveFT(vector, kernel, false);
				//c.convolveJOCL(vector[0], kernel[0]);
				/*
				for (int i = 0; i < vectorWidth; i++) {
					for (int j = 0; j < vectorHeight; j++) {
						for (int k = 0; k < vectorDepth; k++) {
						System.out.format("%.3f ",result[i][j][k]);
						}
					}
					System.out.format("%n");
				}
				*/
			} catch (Throwable t) {
				t.printStackTrace();
			}			
			long time2 = System.currentTimeMillis();
			runTime = (time2-time1)/1000.0;
			if (n >= 5) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
		}

		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-5));
		
		
		
		/*
		System.out.println("UNROLL");
		avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < vectorWidth; i++) {
				for (int j=0; j < vectorHeight; j++) {
					for (int k=0; k < vectorDepth; k++) {
						if (i < 5 && j < 5 && k < 5 ) {
							kernel[i][j][k] = rand.nextDouble();
						}
						vector[i][j][k]= rand.nextDouble();
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result = c.convolveFD(vector, kernel, false);
			long time2 = System.currentTimeMillis();
			double runTime = (time2-time1)/1000.0;
			if (n > 1) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
		}
		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-1));
		/*
		System.out.println("NAIVE");
		avg = 0;
		for (int n = 0; n < trials; n++) {
			for (int i = 0; i < 128; i++) {
				for (int j=0; j < 128; j++) {
					for (int k=0; k < 32; k++) {
						if (i < 5 && j < 5 && k < 5 ) {
							kernel[i][j][k] = rand.nextDouble();
						}
						vector[i][j][k]= rand.nextDouble();
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result = c.convolveFT(vector, kernel, true);
			long time2 = System.currentTimeMillis();
			double runTime = (time2-time1)/1000.0;
			if (n > 1) {
				System.out.print(runTime+" ");
				if (n % 10 == 9) System.out.println();
				avg += runTime;
			}
		}
		System.out.println();
		System.out.format("Average time %.3f %n", avg / (double)(trials-1));
		*/
		
		
		/* FD / FT test
		double[][][] vector = new double[128][128][32];
		double[][][] kernel = new double[5][5][5];
		for (int n = 0; n < 10; n++) {
			for (int i = 0; i < 128; i++) {
				for (int j=0; j<128; j++) {
					for (int k=0; k<32; k++) {
						if (i < 3 && j < 3 && k < 3) {
							kernel[i][j][k] = rand.nextDouble();
						}
						vector[i][j][k] = rand.nextDouble();
					}
				}
			}
			
			long time1 = System.currentTimeMillis();
			double[][][] result = Convolver.convolveFD(vector, kernel);
			long time2 = System.currentTimeMillis();
			System.out.format("Trial %d: time to convolve %.3f seconds %n", n+1, (double)((time2-time1)/1000.0));
		}
		for (int n = 0; n < 10; n++) {
			for (int i = 0; i < 128; i++) {
				for (int j=0; j<128; j++) {
					for (int k=0; k<32; k++) {
						if (i < 3 && j < 3 && k < 3) {
							kernel[i][j][k] = rand.nextDouble();
						}
						vector[i][j][k] = rand.nextDouble();
					}
				}
			}
			Convolver c = new Convolver();
			long time1 = System.currentTimeMillis();
			double[][][] result = Convolver.convolveFT(vector, kernel, false);
			long time2 = System.currentTimeMillis();
			System.out.format("Trial %d: time to convolve %.3f seconds %n", n+1, (double)((time2-time1)/1000.0));
		}
		*/
	}
	
}
