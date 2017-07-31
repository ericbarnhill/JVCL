package com.ericbarnhill.jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

abstract class Convolver<T> {

    public Convolver() {}

    abstract T[] convolve(T[] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[] kernel);
    abstract T[][][] convolve(T[][][] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[][] kernel);
    abstract T[][][] convolve(T[][][] data, T[][] kernel);
    abstract T[][][] convolve(T[][][] data, T[][][] kernel);

     public static String readFile(String fileName) {
        try  {
            BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = null;
            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                sb.append(line).append("\n");
            }
            br.close();
            return sb.toString();
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public static int nextPwr2(int length) {

		int pwr2Length = 1;
		while(pwr2Length < length) {
			pwr2Length *= 2;
		}
		return pwr2Length;
    }
}
