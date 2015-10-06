package jvcl;

public class DimShift {
	
	public DimShift() {};

	public double[][] shiftDim(double[][] image) {
		int width = image.length;
		int height = image[0].length;
		double[][] result = new double[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[x][y] = image[y][x];
			}
		}
		return result;
	}

	public float[][] shiftDim(float[][] image) {
		int width = image.length;
		int height = image[0].length;
		float[][] result = new float[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[x][y] = image[y][x];
			}
		}
		return result;
	}

	public double[][][] shiftDim(double[][][] image) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		double[][][] result = new double[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					result[x][y][z] = image[y][z][x];
				}
			}
		}
		return result;
	}
	
	public float[][][] shiftDim(float[][][] image) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		float[][][] result = new float[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					result[x][y][z] = image[y][z][x];
				}
			}
		}
		return result;
	}
	
	public double[][][] shiftDim(double[][][] image, int nDims) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		nDims = nDims % 3;
		double[][][] result;
		if (nDims == 1) {
			result = new double[height][depth][width];
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < depth; y++) {
					for (int z = 0; z < width; z++) {
						result[x][y][z] = image[y][z][x];
					}
				}
			}
		} else if (nDims == 2) {
			result = new double[depth][width][height];
			for (int x = 0; x < depth; x++) {
				for (int y = 0; y < width; y++) {
					for (int z = 0; z < height; z++) {
						result[x][y][z] = image[z][x][y];
					}
				}
			}
		} else return image;
		return result;
	}	
	

	public float[][][] shiftDim(float[][][] image, int nDims) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		nDims = nDims % 3;
		float[][][] result;
		if (nDims == 1) {
			result = new float[height][depth][width];
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < depth; y++) {
					for (int z = 0; z < width; z++) {
						result[x][y][z] = image[y][z][x];
					}
				}
			}
		} else if (nDims == 2) {
			result = new float[depth][width][height];
			for (int x = 0; x < depth; x++) {
				for (int y = 0; y < width; y++) {
					for (int z = 0; z < height; z++) {
						result[x][y][z] = image[z][x][y];
					}
				}
			}
		} else return image;
		return result;
	}	

	
	

}
