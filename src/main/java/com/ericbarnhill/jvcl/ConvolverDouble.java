package com.ericbarnhill.jvcl;

abstract class ConvolverDouble extends Convolver<Double> {

    public ConvolverDouble() {
        super();
    }

    abstract double[] convolve(double[] f, double[] g);
    abstract double[][] convolve(double[][] f, double[] g);
    abstract double[][][] convolve(double[][][] f, double[] g);
    abstract double[][] convolve(double[][] f, double[][] g);
    abstract double[][][] convolve(double[][][] f, double[][] g);
    abstract double[][][] convolve(double[][][] f, double[][][] g);

    abstract Double[] convolve(Double[] f, Double[] g);
    abstract Double[][] convolve(Double[][] f, Double[] g);
    abstract Double[][][] convolve(Double[][][] f, Double[] g);
    abstract Double[][] convolve(Double[][] f, Double[][] g);
    abstract Double[][][] convolve(Double[][][] f, Double[][] g);
    abstract Double[][][] convolve(Double[][][] f, Double[][][] g);
}
