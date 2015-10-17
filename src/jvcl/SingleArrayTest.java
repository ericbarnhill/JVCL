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

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLProgram;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import java.beans.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.prefs.Preferences;

import jvcl.ProgressBar.Task;

public class SingleArrayTest {

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
    private final int arrayMin = 64;
    private final int arrayMax = 1024;
    private final int kernelMin = 3;
    private final int kernelMax = 1024;
    Random random;
    final int NAIVE = 0;
    final int UNROLLED = 1;
    final int FTCPU = 2;
    final int FDGPU = 3;
    final int FTGPU = 4;
    final String[] bestNames = {"Naive FD", "Unrolled FD", "FT on CPU", "FD on GPU", "FT on GPU"};
    Preferences p;
    byte[] convolverPrefs;
    long[] temp;
    boolean hasOpenCL;
    int progress;
    int arraySize, kernelSize;
    
    
	public SingleArrayTest() {
		random = new Random();
		//convolverPrefs = new byte[numArraySizes*numKernelSizes];
		temp = new long[5];
	}

	// check for openCL
	// run all tests using: FDCPUNaive, FDGPU, FTCPU, FTGPU
	// also small kernel tests with FDCPUUnrolled

	/** runs the SingleArrayTest */
	public void tune() {
		
		System.out.println();
		File file = new File("/home/ericbarnhill/Documents/code/results.csv");
		file.delete();
		pb = new ProgressBar();
		progress = 0;
		pb.createStatusBar();
		pb.taskOutput.append(String.format("JVCL SingleArrayTest. Press Start to Begin. \n"));
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
			//ftgpu.setDebug(true);
		}
		try {
			pb.startProgress();
			//run1dTests();
			run2dTests();
			//run3dTests();
			//writePreferences();
		} finally {
			fdgpu.close();
			ftgpu.close();
		}
		return;
	}
	
	private void run1dTests() {
		long start, end, duration;
		for (arraySize = arrayMin; arraySize < arrayMax; arraySize += 32) {
			for (kernelSize = kernelMin; kernelSize < arraySize; kernelSize = kernelSize * 2 - 1) {
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d \n", arraySize, kernelSize));
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
				pb.taskOutput.append(String.format("1D Naive FD: %.3f sec %n", duration / 1000.0 ));
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
					pb.taskOutput.append(String.format("1D Unrolled FD: %.3f sec %n", duration / 1000.0 ));
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
				pb.taskOutput.append(String.format("1D FT on CPU: %.3f sec %n", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("1D FD on GPU: %.3f sec %n", duration / 1000.0 ));
					temp[FDGPU] = duration;
					// FTGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolve(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("1D FT on GPU: %.3f sec %n", duration / 1000.0 ));
					temp[FTGPU] = duration;
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				} 
				byte min = min();
				//convolverPrefs[m*numKernelSizes + n] = min;
				pb.taskOutput.append(String.format("Best Performer:  %s %n", bestNames[min]));
				writeResults();
				//float progress = 100*(m*numKernelSizes + n) / ((float)totalTests*3.0f);
				pb.setProgress((int)(100*(arraySize/arrayMax)*(kernelSize/kernelMax) / 3.0));
			} // for kernel size
		} // for array size
		
	}
	
	private void run2dTests() {
		long start, end, duration;
		arraySize = 512;
		int kernelSize = 3;
			while (kernelSize < arraySize) {
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d \n", arraySize, kernelSize));
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
				pb.taskOutput.append(String.format("2D Naive FD: %.3f sec %n", duration / 1000.0 ));
				temp[NAIVE] = duration;
				System.gc();
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
					pb.taskOutput.append(String.format("2D Unrolled FD: %.3f sec %n", duration / 1000.0 ));
					temp[UNROLLED] = duration;
				} else {
					temp[UNROLLED] = Integer.MAX_VALUE;
				}
				System.gc();
				// FTCPU
				start = System.currentTimeMillis();
				for (int t = 0; t < 10; t++) {
					ftcpu.convolve(array, kernel, false);
				}
				end = System.currentTimeMillis();
				duration = end-start;
				pb.taskOutput.append(String.format("2D FT on CPU: %.3f sec %n", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("2D FD on GPU: %.3f sec %n", duration / 1000.0 ));
					temp[FDGPU] = duration;
					System.gc();
					// FTGPU STRIDE
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolveStride(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("2D FT on GPU STRIDE: %.3f sec %n", duration / 1000.0 ));
					temp[FTGPU] = duration;
					System.gc();
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				} 
				writeResults();
				byte min = min();
				//convolverPrefs[m*numKernelSizes + n] = min;
				pb.taskOutput.append(String.format("Best Performer:  %s %n", bestNames[min]));
				//float progress = 2*100*(m*numKernelSizes + n) / ((float)totalTests*3.0f);
				pb.setProgress((int)(33.3 + (arraySize/arrayMax)*(kernelSize/kernelMax) / 3.0));
				kernelSize = kernelSize + 2;
				if (kernelSize > 64) {
					kernelSize = kernelSize + 2;
				}
				if (kernelSize > 128) {
					kernelSize = kernelSize + 4;
				}
				if (kernelSize > 256) {
					kernelSize = kernelSize + 8;
				}
			} // while kernel size
	}

	private void run3dTests() {
		long start, end, duration;
		for (int arraySize = arrayMin; arraySize < arrayMax; arraySize += 32) {
			for (int kernelSize = kernelMin; kernelSize < kernelMax; kernelSize = kernelSize * 2 - 1) {
				pb.taskOutput.append(String.format("Array size %d Kernel Size %d \n", arraySize, kernelSize));
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
				pb.taskOutput.append(String.format("3D Naive FD: %.3f sec %n", duration / 1000.0 ));
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
					pb.taskOutput.append(String.format("3D Unrolled FD: %.3f sec %n", duration / 1000.0 ));
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
				pb.taskOutput.append(String.format("3D FT on CPU: %.3f sec %n", duration / 1000.0 ));
				temp[FTCPU] = duration;
				if (hasOpenCL) {
					// FDGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						fdgpu.convolve(array, kernel);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("3D FD on GPU: %.3f sec %n", duration / 1000.0 ));
					temp[FDGPU] = duration;
					// FTGPU
					start = System.currentTimeMillis();
					for (int t = 0; t < 10; t++) {
						ftgpu.convolve(array, kernel, false);
					}
					end = System.currentTimeMillis();
					duration = end-start;
					pb.taskOutput.append(String.format("3D FT on GPU: %.3f sec %n", duration / 1000.0 ));
					temp[FTGPU] = duration;
				} else {
					temp[FTGPU] = Integer.MAX_VALUE;
					temp[FDGPU] = Integer.MAX_VALUE;
				}
				writeResults();
				byte min = min();
				//convolverPrefs[m*numKernelSizes + n] = min;
				pb.taskOutput.append(String.format("Best Performer:  %s %n", bestNames[min]));
				//float progress = 2*100*(m*numKernelSizes + n) / ((float)totalTests*3.0f);
				pb.setProgress((int)(66.6 + 100*(arraySize/arrayMax)*(kernelSize/kernelMax) / 3.0f));
			} // for kernel size
		} // for array size
	}
	
	byte min() {
		double min = Integer.MAX_VALUE;
		byte minIndex = 0;
		for (byte n = 0; n < 5; n++) {
			if (temp[n] < min) {
				min = temp[n];
				minIndex = n;
			}
		}
		return minIndex;
	}
	
	void writePreferences() {
		p = Preferences.userNodeForPackage( this.getClass() );
		p.putByteArray("convolverPrefs", convolverPrefs);		
	}
	
	void writeResults() {
		BufferedWriter br;
		try {
			br = new BufferedWriter(new FileWriter("/home/ericbarnhill/Documents/code/results.csv", true));
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("%d, %d ", arraySize, kernelSize));
			for (int x = 0; x < 5; x++) {
				if (temp[x] != Integer.MAX_VALUE) {
					sb.append(String.format("%d, ", temp[x]));
				} else {
					sb.append(String.format("%d, ", -1));
				}
			}
			sb.append(String.format("%n"));
			br.write(sb.toString());	
			br.close();
		} catch (IOException e) {
			System.out.println("IO Exception");
		} 
	
	}
	
	
	/** Main method for SingleArrayTest */
	public static void main(String[] args) {
		new SingleArrayTest().tune();
	}



}
	
	
	
