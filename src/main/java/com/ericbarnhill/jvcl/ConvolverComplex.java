package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;

abstract class ConvolverComplex extends Convolver<Complex> {

    public ConvolverComplex() {
            super();
    }

    abstract Complex[] convolve(Complex[] f, Complex[] g);
    abstract Complex[][] convolve(Complex[][] f, Complex[] g);
    abstract Complex[][][] convolve(Complex[][][] f, Complex[] g);
    abstract Complex[][] convolve(Complex[][] f, Complex[][] g);
    abstract Complex[][][] convolve(Complex[][][] f, Complex[][] g);
    abstract Complex[][][] convolve(Complex[][][] f, Complex[][][] g);

}
