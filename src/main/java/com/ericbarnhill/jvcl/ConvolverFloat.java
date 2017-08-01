package com.ericbarnhill.jvcl;

abstract class ConvolverFloat extends Convolver<Float> {

    public ConvolverFloat() {
        super();
    }

    abstract Float[] convolve(Float[] f, Float[] g);
    abstract Float[][] convolve(Float[][] f, Float[] g);
    abstract Float[][][] convolve(Float[][][] f, Float[] g);
    abstract Float[][] convolve(Float[][] f, Float[][] g);
    abstract Float[][][] convolve(Float[][][] f, Float[][] g);
    abstract Float[][][] convolve(Float[][][] f, Float[][][] g);


}
