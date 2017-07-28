package com.ericbarnhill.jvcl;

abstract class Convolver<T> {

    public Convolver() {}

    abstract T[] convolve(T[] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[] kernel);
    abstract T[][][] convolve(T[][][] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[][] kernel);
    abstract T[][] convolve(T[][][] data, T[][] kernel);
    abstract T[][][] convolve(T[][][] data, T[][][] kernel);

}
