package jvcl;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;
import org.apache.commons.math4.util.FastMath;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;

public class FFTTest {

	public static void main(String[] args) {
		StockhamGPUStride s = new StockhamGPUStride();
		for (int n = 5; n < 11; n++) {
			int dim = (int)FastMath.pow(2, n);
			System.out.println("Dim "+dim);
			Complex[][] array = JVCLUtils.initializeRandom(dim);
			DoubleFFT_2D f = new DoubleFFT_2D(dim, dim);
			System.gc();
			long time1 = System.currentTimeMillis();
			for (int t = 0; t < 100; t++) {
				f.complexForward(ComplexUtils.complex2Interleaved(array));
			}
			long time2 = System.currentTimeMillis();
			System.out.format("CPU %.6f %n",(time2-time1)/1000.0);
			long time3 = System.currentTimeMillis();
			for (int t = 0; t < 100; t++) {
				s.fft(array, true, dim);
			}
			long time4 = System.currentTimeMillis();
			System.out.format("GPU %.6f %n",(time4-time3)/1000.0);
		}
	}

}
