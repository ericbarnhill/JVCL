package com.ericbarnhill.jvcl;

//NOTA BENE: Always convert complex to interleaved when in vector format if possible. Saves all kind of headaches.

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static java.lang.Math.min;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class FDGPU{
	
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program1d, program2d, program3d, program1dComplex, program2dComplex, program3dComplex,
    	program21, program21Complex, program31, program31Complex;
    CLContext context;
    int localWorkSize;
	
	public FDGPU() {
		//try {
			context = CLContext.create();
			device = context.getMaxFlopsDevice();
	        queue = device.createCommandQueue();
	        Path currentRelativePath = Paths.get("");
	        String s = currentRelativePath.toAbsolutePath().toString();
	        System.out.println("Current relative path is: " + s);
	        String path="/home/ericbarnhill/barnhill-eclipse-workspace/JVCL/";
	        String source1d = JVCLUtils.readFile(path+"src/Convolve1d.cl");
	        String source2d = JVCLUtils.readFile(path+"src/Convolve2d.cl");
	        String source3d = JVCLUtils.readFile(path+"src/Convolve3d.cl");
	        String source1dComplex = JVCLUtils.readFile(path+"src/Convolve1dComplex.cl");
	        String source2dComplex = JVCLUtils.readFile(path+"src/Convolve2dComplex.cl");
	        String source3dComplex = JVCLUtils.readFile(path+"src/Convolve3dComplex.cl");
	        String source21 = JVCLUtils.readFile(path+"src/Convolve21.cl");
	        String source21Complex = JVCLUtils.readFile(path+"src/Convolve21Complex.cl");
	        String source31 = JVCLUtils.readFile(path+"src/Convolve31.cl");
	        String source31Complex = JVCLUtils.readFile(path+"src/Convolve31Complex.cl");
	        program1d = context.createProgram(source1d).build(); 
	        program2d = context.createProgram(source2d).build(); 
	        program3d = context.createProgram(source3d).build(); 
	        program21 = context.createProgram(source21).build();
	        program31 = context.createProgram(source31).build();
	        program1dComplex = context.createProgram(source1dComplex).build(); 
	        program2dComplex = context.createProgram(source2dComplex).build(); 
	        program3dComplex = context.createProgram(source3dComplex).build(); 
	        program21Complex = context.createProgram(source21Complex).build();
	        program31Complex = context.createProgram(source31Complex).build();
			localWorkSize = min(device.getMaxWorkGroupSize(), 32);  // Local work size dimensions
		//} catch (Exception e) {
		//	throw new NoGPUException(); 
		//}
	}

	public double[] convolve(double[] f, double[] g) {
		final int fi = f.length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri, WRITE_ONLY);
        clF.getBuffer().put(JVCLUtils.double2Float(fPad)).rewind();
        clG.getBuffer().put(JVCLUtils.double2Float(g)).rewind();
        CLKernel Kernel = program1d.createCLKernel("Convolve1d");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put1DRangeKernel(Kernel, 0, JVCLUtils.roundUp(ri, localWorkSize),0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri];
		clR.getBuffer().get(result);  
		clF.release();
		clG.release();
		clR.release();
        return JVCLUtils.float2Double(result);    		
	}
	

	public Complex[] convolve(Complex[] f, Complex[] g) {
		final int fi = f.length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(fPad)).rewind();
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program1dComplex.createCLKernel("Convolve1dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put1DRangeKernel(Kernel, 0, JVCLUtils.roundUp(ri, localWorkSize),0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*2];
		clR.getBuffer().get(result);
		clF.release();
		clG.release();
		clR.release();
        return ComplexUtils.interleaved2Complex(result);   
		   		
	}
	

	public double[][] convolve(double[][] f, double[] g, int dim) {
		if (dim == 1) f = ArrayMath.shiftDim(f);
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, 0, 0);
		double[][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj, WRITE_ONLY);
        clF.getBuffer().put(JVCLUtils.double2Float(ArrayMath.vectorise(fPad))).rewind();
        clG.getBuffer().put(JVCLUtils.double2Float(g)).rewind();
        CLKernel Kernel = program21.createCLKernel("Convolve21");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, fj,0 ,0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj];
		clR.getBuffer().get(resultVec); 
		clF.release();
		clG.release();
		clR.release();
        double[][] result = ArrayMath.devectorise(JVCLUtils.float2Double(resultVec), ri); 
		// if (dim == 1) result = JVCLUtils.shiftDim(result);
		return result;
	}
	
	public double[][] convolve(double[][] f, double[] g) {
		return convolve(f, g, 0);
	}
	
	public Complex[][] convolve(Complex[][] f, Complex[] g, int dim) {
		if (dim == 1) {
			f = ArrayMath.shiftDim(f);
		}
        JVCLUtils.display(ArrayMath.vectorise(f), "21c f", f.length);
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, 0, 0);
		Complex[][] r = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(ArrayMath.vectorise(fPad))).rewind();
        JVCLUtils.display(ComplexUtils.complex2InterleavedFloat(ArrayMath.vectorise(fPad)), "21c fpad", ri*2);
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program21Complex.createCLKernel("Convolve21Complex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, fj, 0,0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj*2];
		clR.getBuffer().get(resultVec);  
		clF.release();
		clG.release();
		clR.release();
		Complex[][] result = ArrayMath.devectorise(ComplexUtils.interleaved2Complex(resultVec), ri); 
		if (dim == 1) result = ArrayMath.shiftDim(result);
        return result;
	}
	
	public Complex[][] convolve(Complex[][] f, Complex[] g) {
		return convolve(f, g, 0);
	}
	
	public double[][][] convolve(double[][][] f, double[] g, int dim) {
		if (dim == 1) f = ArrayMath.shiftDim(f, 1);
		if (dim == 2) f = ArrayMath.shiftDim(f, 2);
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, 0, 0, 0, 0);
		double[][][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, 0, 0, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj*fk, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj*fk, WRITE_ONLY);
        clF.getBuffer().put(JVCLUtils.double2Float(ArrayMath.vectorise(fPad))).rewind();
        clG.getBuffer().put(JVCLUtils.double2Float(g)).rewind();
        CLKernel Kernel = program31.createCLKernel("Convolve31");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, fj, fk, 0,0, 0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj*fk];
		clR.getBuffer().get(resultVec);  
		clF.release();
		clG.release();
		clR.release();
        double[][][] result = ArrayMath.devectorise(JVCLUtils.float2Double(resultVec), ri, fj);    
        if (dim == 1) result = ArrayMath.shiftDim(result, 2);
		if (dim == 2) result = ArrayMath.shiftDim(result, 1);
		return result;
	}
	
	public double[][][] convolve(double[][][] f, double[] g) {
		return convolve(f, g, 0);
	}
	

	public Complex[][][] convolve(Complex[][][] f, Complex[] g, int dim) {
		if (dim == 1) f = ArrayMath.shiftDim(f, 1);
		if (dim == 2) f = ArrayMath.shiftDim(f, 2);
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, 0, 0, 0, 0);
		Complex[][][] r = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, 0, 0, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj*fk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj*fk*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(ArrayMath.vectorise(fPad))).rewind();
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program31Complex.createCLKernel("Convolve31Complex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, fj, fk, 0,0,0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj*fk*2];
		clR.getBuffer().get(resultVec);  
		clF.release();
		clG.release();
		clR.release();
        Complex[][][] result = ArrayMath.devectorise(ComplexUtils.interleaved2Complex(resultVec), ri, fj);    
        if (dim == 1) result = ArrayMath.shiftDim(result, 2);
		if (dim == 2) result = ArrayMath.shiftDim(result, 1);
		return result;
	}
	
	public Complex[][][] convolve(Complex[][][] f, Complex[] g) {
		return convolve(f, g, 0);
	}

	
	
	public double[][] convolve(double[][] f, double[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		double[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
		double[][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
		final int ri = r.length;
		final int rj = r[0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorise(JVCLUtils.double2Float(fPad)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorise(JVCLUtils.double2Float(g)))
    		.rewind();
        CLKernel Kernel = program2d.createCLKernel("Convolve2d");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgie)
        	.putArg(hgje);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, rj, 0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj];
		clR.getBuffer().get(result);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorise(JVCLUtils.float2Double(result), ri);
	}
	

	public Complex[][] convolve(Complex[][] f, Complex[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		Complex[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
		Complex[][] r = JVCLUtils.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj]),
				hgi, hgie, hgj, hgje);
		final int ri = r.length;
		final int rj = r[0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj*2, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorise(ComplexUtils.complex2InterleavedFloat(fPad, 0)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorise(ComplexUtils.complex2InterleavedFloat(g, 0)))
    		.rewind();
        CLKernel Kernel = program2dComplex.createCLKernel("Convolve2dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgie)
        	.putArg(hgje);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, rj, 0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj*2];
		clR.getBuffer().get(result);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorise(ComplexUtils.interleaved2Complex(result), ri);
	}

	
	public double[][][] convolve(double[][][] f, double[][][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgk = (int)( (gk - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
		double[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
		double[][][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
		final int ri = r.length;
		final int rj = r[0].length;
		final int rk = r[0][0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj*rk, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj*gk, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj*rk, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorise(JVCLUtils.double2Float(fPad)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorise(JVCLUtils.double2Float(g)))
    		.rewind();
        CLKernel Kernel = program3d.createCLKernel("Convolve3d");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(rk)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(gk)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgk)
        	.putArg(hgie)
        	.putArg(hgje)
        	.putArg(hgke);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, rj, rk, 0,0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj*rk];
		clR.getBuffer().get(result);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorise(JVCLUtils.float2Double(result), ri, rj);
	}

	public Complex[][][] convolve(Complex[][][] f, Complex[][][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgk = (int)( (gk - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
		//Complex[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
		Complex[][][] fPad = JVCLUtils.zeroPadBoundaries(f, 1);
		JVCLUtils.display(ArrayMath.vectorise(ComplexUtils.complex2InterleavedFloat(fPad, 0)), "3D Complex FPad", fPad.length*2);
		//Complex[][][] r = JVCLUtils.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj][fk]),
		//		hgi, hgie, hgj, hgje, hgk, hgke);
		Complex[][][] r = JVCLUtils.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj][fk]),1);
		final int ri = r.length;
		final int rj = r[0].length;
		final int rk = r[0][0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj*rk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj*gk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj*rk*2, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorise(ComplexUtils.complex2InterleavedFloat(fPad, 0)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorise(ComplexUtils.complex2InterleavedFloat(g, 0)))
    		.rewind();
        CLKernel Kernel = program3dComplex.createCLKernel("Convolve3dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(rk)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(gk)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgk)
        	.putArg(hgie)
        	.putArg(hgje)
        	.putArg(hgke);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, rj, rk, 0,0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj*rk*2];
		clR.getBuffer().get(result);
		JVCLUtils.display(result, "3D Complex Result", fPad.length*2);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorise(ComplexUtils.interleaved2Complex(result), ri, rj);
	}

	
	public void close() {
        context.release();
	}

	public static void main(String[] args) {
		
		FDGPU fdgpu = new FDGPU();
		try {
			double[][][] test = new double[16][16][16];
			Complex[][][] testC = new Complex[16][16][16];
			for (int n = 0; n < 16; n++) {
				for (int p = 0; p < 16; p++) {
					test[n][p] = JVCLUtils.fillWithSecondOrder(16);
					testC[n][p] = ComplexUtils.real2Complex(JVCLUtils.fillWithSecondOrder(16));
				}
			}
			//JVCLUtils.display(ArrayMath.vectorise(testC), "vec 21C", testC.length);
			//double[][] convolve21 = fdgpu.convolve(JVCLUtils.shiftDim(test), new double[] {1, -2, 1}, 0);
			double[][][] convolve31 = fdgpu.convolve(test, new double[] {1, -2, 1}, 2);
			JVCLUtils.display(ArrayMath.vectorise(convolve31), "convolve 31", convolve31.length);
			Complex[][][] convolve31C = fdgpu.convolve(testC, ComplexUtils.real2Complex(new double[] {1, -2, 1}), 2);
			JVCLUtils.display(ArrayMath.vectorise(convolve31C), "convolve 31C", convolve31C.length);
			
			/*
			double[] f1 = JVCLUtils.fillWithSecondOrder(16);
			double[] g1 = new double[] {1, -2, 1};
			JVCLUtils.display(fdgpu.convolve(f1, g1), "1D Double", 16);
			Complex[] f2 = JVCLUtils.fillWithSecondOrderComplex(16);
			Complex[] g2 = ComplexUtils.real2Complex(g1);
			JVCLUtils.display(fdgpu.convolve(f2, g2), "1D Complex", 16);
			double[][] f3 = JVCLUtils.fillWithSecondOrder(16, 16);
			JVCLUtils.display(ArrayMath.vectorise(f3), "Second Order", f3.length);
			double[][] g3 = ArrayMath.devectorise(new double[] {0,1,0, 1,-4,1,0,1,0,}, 3);
			JVCLUtils.display(ArrayMath.vectorise(g3), "Second Order Kernel", g3.length);
			double[][] r3 = fdgpu.convolve(f3, g3);
			JVCLUtils.display(ArrayMath.vectorise(r3), "2D Double", r3.length);
			Complex[][] f4 = JVCLUtils.fillWithSecondOrderComplex(16, 16);
			Complex[][] g4 = ComplexUtils.real2Complex(g3);
			Complex[][] r4 = fdgpu.convolve(f4, g4);
			JVCLUtils.display(ArrayMath.vectorise(r4), "2D Complex", r4.length);
			/*
			double[][][] f5 = JVCLUtils.fillWithSecondOrder(8, 8, 8);
			
			double[][][] g5 = ArrayMath.devectorise(new double[] {
					0,0,0,0,1,0,0,0,0,0,1,0,1,-6,1,0,1,0,0,0,0,0,1,0,0,0,0}, 3, 3, 0);
			JVCLUtils.display(ArrayMath.vectorise(g5, 0), "Third Order Kernel", g5.length);
			//double[][][] r5 = fdgpu.convolve(f5, g5);
			//JVCLUtils.display(ArrayMath.vectorise(r5, 0), "3D Double", r5.length);
			Complex[][][] f6 = JVCLUtils.fillWithSecondOrderComplex(8, 8, 8);
			Complex[][][] g6 = ComplexUtils.real2Complex(g5);
			Complex[][][] r6 = fdgpu.convolve(f6, g6);
			JVCLUtils.display(ArrayMath.vectorise(r6, 0), "3D Complex", r6.length);
			double[] test = JVCLUtils.fillWithGradient(27);
			//double[][] test2 = ArrayMath.devectorise(test, 4, 0)
			//JVCLUtils.display(test, "test", 4);
			*/
			
		} finally {
			fdgpu.close();
		}
	}

}
