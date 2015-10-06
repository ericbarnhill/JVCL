package jvcl;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLProgram;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import java.beans.*;
import java.util.Random;

import jvcl.ProgressBar.Task;

public class Autotuner {

	FDCPUNaive naive;
	FDCPUUnrolled unrolled;
	FDGPU fdgpu;
	FTCPU ftcpu;
	FTGPU ftgpu;
	ProgressBar pb;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
    private final int[] arraySizes = {64, 128, 256, 512, 1024};
    private final int[] kernelSizes = {3, 5, 7, 9, 11, 13, 15};
    private final int numArraySizes = 5;
    private final int numKernelSizes = 7;
    Random random;
    final int NAIVE = 0;
    final int UNROLLED = 1;
    final int FTCPU = 2;
    final int FDGPU = 3;
    final int FTGPU = 4;
    final String[] bestNames = {"Naive FD", "Unrolled FD", "FT on CPU", "FD on GPU", "FT on GPU"};
    int[][] best;
    double[] temp;
    boolean hasOpenCL;
    
    
	public Autotuner() {
		random = new Random();
		best = new int[numArraySizes][numKernelSizes];
		temp = new double[5];
	}

	// check for openCL
	// run all tests using: FDCPUNaive, FDGPU, FTCPU, FTGPU
	// also small kernel tests with FDCPUUnrolled

	public void tune() {
		
		System.out.println();
		pb = new ProgressBar();
		pb.createStatusBar();
		pb.taskOutput.append(String.format("JVCL Autotuner. Press Start to Begin. %n"));
		hasOpenCL = true;
		try {
			context = CLContext.create();
			device = context.getMaxFlopsDevice();
	        queue = device.createCommandQueue();
		} catch (com.jogamp.opencl.CLException e) {
			hasOpenCL = false;
			pb.taskOutput.append("No OpenCL Found. Skipping OpenCL tests...");
		}
		while(!pb.started) {
			try { Thread.sleep(100); } catch (InterruptedException ignore) {}
		}
		naive = new FDCPUNaive();
		unrolled = new FDCPUUnrolled();
		ftcpu = new FTCPU();
		if (hasOpenCL) {
			fdgpu = new FDGPU();
			ftgpu = new FTGPU();
		}
		run1dTests();
		run2dTests();
		run3dTests();
		return;
	}
	
	private void run1dTests() {
		long start, end, duration;
		for (int m = 0; m < numArraySizes; m++) {
			for (int n = 0; n < numKernelSizes; n++) {
				int arraySize = arraySizes[m];
				int kernelSize = kernelSizes[n];
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d %n", arraySize, kernelSize));
				double[] array = new double[arraySize];
				double[] kernel = new double[kernelSize];
				for (int x = 0; x < arraySize; x++) {
					if (x < kernelSize) {
						kernel[x] = random.nextDouble();
					}
					array[x] = random.nextDouble();
				}
				// FD Naive
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					naive.convolve(array, kernel);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("Naive FD: %d sec", duration / 1000.0 ));
				temp[NAIVE] = duration;
				if (kernel.length <= 5) {
					// FD Unrolled
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						if (kernel.length == 3) {
							unrolled.convolve3(array, kernel);
						} else {
							unrolled.convolve5(array,kernel);
						}
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("Unrolled FD: %d sec", duration / 1000.0 ));
					temp[UNROLLED] = duration;
				} else {
					temp[UNROLLED] = Integer.MAX_VALUE;
				}
				// FTCPU
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					ftcpu.convolve(array, kernel, false);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("FT on CPU: %d sec", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FD on GPU: %d sec", duration / 1000.0 ));
					temp[FDGPU] = duration;
					// FTGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolve(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FT on GPU: %d sec", duration / 1000.0 ));
					temp[FTGPU] = duration;
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				} 
				int min = min();
				best[m][n] = min;
				pb.taskOutput.append(String.format("Best Performer: ", bestNames[min]));
			} // for kernel size
		} // for array size
	}
	
	private void run2dTests() {
		long start, end, duration;
		for (int m = 0; m < numArraySizes; m++) {
			for (int n = 0; n < numKernelSizes; n++) {
				int arraySize = arraySizes[m];
				int kernelSize = kernelSizes[n];
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d %n", arraySize, kernelSize));
				double[][] array = new double[arraySize][arraySize];
				double[][] kernel = new double[kernelSize][kernelSize];
				for (int x = 0; x < arraySize; x++) {
					for (int y = 0; y < arraySize; y++) {
						if (x < kernelSize && y < kernelSize) {
							kernel[x][y] = random.nextDouble();
						}
						array[x][y] = random.nextDouble();
					}
				}
				// FD Naive
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					naive.convolve(array, kernel);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("Naive FD: %d sec", duration / 1000.0 ));
				temp[NAIVE] = duration;
				if (kernel.length <= 5) {
					// FD Unrolled
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						if (kernel.length == 3) {
							unrolled.convolve3(array, kernel);
						} else {
							unrolled.convolve5(array,kernel);
						}
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("Unrolled FD: %d sec", duration / 1000.0 ));
					temp[UNROLLED] = duration;
				} else {
					temp[UNROLLED] = Integer.MAX_VALUE;
				}
				// FTCPU
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					ftcpu.convolve(array, kernel, false);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("FT on CPU: %d sec", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FD on GPU: %d sec", duration / 1000.0 ));
					temp[FDGPU] = duration;
					// FTGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolve(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FT on GPU: %d sec", duration / 1000.0 ));
					temp[FTGPU] = duration;
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				} 
				best[m][n] = min();
			} // for kernel size
		} // for array size
	}

	private void run3dTests() {
		long start, end, duration;
		for (int m = 0; m < numArraySizes-1; m++) { // 1024^3 may be too much
			for (int n = 0; n < numKernelSizes; n++) {
				int arraySize = arraySizes[m];
				int kernelSize = kernelSizes[n];
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d %n", arraySize, kernelSize));
				double[][][] array = new double[arraySize][arraySize][arraySize];
				double[][][] kernel = new double[kernelSize][kernelSize][kernelSize];
				for (int x = 0; x < arraySize; x++) {
					for (int y = 0; y < arraySize; y++) {
						for (int z = 0; z < arraySize; z++) {
							if (x < kernelSize && y < kernelSize && z < kernelSize) {
								kernel[x][y][z] = random.nextDouble();
							}
							array[x][y][z] = random.nextDouble();
						}
					}
				}
				// FD Naive
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					naive.convolve(array, kernel);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("Naive FD: %d sec", duration / 1000.0 ));
				temp[NAIVE] = duration;
				if (kernel.length <= 5) {
					// FD Unrolled
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						if (kernel.length == 3) {
							unrolled.convolve3(array, kernel);
						} else {
							unrolled.convolve5(array,kernel);
						}
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("Unrolled FD: %d sec", duration / 1000.0 ));
					temp[UNROLLED] = duration;
				} else {
					temp[UNROLLED] = Integer.MAX_VALUE;
				}
				// FTCPU
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					ftcpu.convolve(array, kernel, false);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("FT on CPU: %d sec", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FD on GPU: %d sec", duration / 1000.0 ));
					temp[FDGPU] = duration;
					// FTGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolve(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("FT on GPU: %d sec", duration / 1000.0 ));
					temp[FTGPU] = duration;
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				}
				best[m][n] = min();
			} // for kernel size
		} // for array size
	}
	
	int min() {
		int min = Integer.MAX_VALUE;
		int minIndex = 0;
		for (int n = 0; n < 5; n++) {
			if (temp[n] < min) minIndex = n;
		}
		return minIndex;
	}
	
	public static void main(String[] args) {
		new Autotuner().tune();
	}

}
	
	
	
