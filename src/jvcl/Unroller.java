package jvcl;

public class Unroller {
	
	public static void makeUnroll(int xDim, int yDim, int zDim) {
		
		int halfWidth = xDim/2;
		int halfHeight = yDim/2;
		int halfDepth = zDim/2;
		String xString = ""; String yString = ""; String zString = "";
		System.out.println("result[x][y][z] = ");
		for (int x = -halfWidth; x <= halfWidth; x++) {
			for (int y = -halfHeight; y <= halfHeight; y++) {
				for (int z = -halfDepth; z <= halfDepth; z++) {
					if (x>=0) {
						xString = String.format("[x+%d]",x);
					} else {
						xString = String.format("[x%d]",x);
					}
					if (y>=0) {
						yString = String.format("[y+%d]",y);
					} else {
						yString = String.format("[y%d]",y);
					}
					if (z>=0) {
						zString = String.format("[z+%d]",z);
					} else {
						zString = String.format("[z%d]",z);
					}
					//System.out.format("volume%s%s%s*kernel[%d][%d][%d]",xString,yString,zString, x, y, z);
					System.out.format("volume%s%s*kernel[%d][%d]",xString,yString, x, y);
					if (x == halfWidth && y == halfHeight && z == halfDepth) {
						System.out.format(";%n");
					} else {
						System.out.format(" + %n");
					}
				}
			}
		}
	}
	

	public static void makeUnrollComplex(int xDim, int yDim) {
		int halfWidth = xDim/2;
		int halfHeight = yDim/2;
		String xString = ""; String yString = ""; String zString = "";
		System.out.println("result[x][y]");
		for (int x = -halfWidth; x <= halfWidth; x++) {
			for (int y = -halfHeight; y <= halfHeight; y++) {
				if (x>=0) {
					xString = String.format("[x+%d]",x);
				} else {
					xString = String.format("[x%d]",x);
				}
				if (y>=0) {
					yString = String.format("[y+%d]",y);
				} else {
					yString = String.format("[y%d]",y);
				}
				//System.out.format("volume%s%s%s*kernel[%d][%d][%d]",xString,yString,zString, x, y, z);
				System.out.format(".add(image%s%s.multiply(kernel[%d][%d]))",xString,yString, x, y);
				if (x == halfWidth && y == halfHeight) {
					System.out.format(";%n");
				} else {
					System.out.format("%n");
				}
			}
		}
	}

	public static void makeUnrollComplex(int xDim, int yDim, int zDim) {
		int halfWidth = xDim/2;
		int halfHeight = yDim/2;
		int halfDepth = zDim/2;
		String xString = ""; String yString = ""; String zString = "";
		System.out.println("result[x][y][z]");
		for (int x = -halfWidth; x <= halfWidth; x++) {
			for (int y = -halfHeight; y <= halfHeight; y++) {
				for (int z = -halfDepth; z <= halfDepth; z++) {
					if (x>=0) {
						xString = String.format("[x+%d]",x);
					} else {
						xString = String.format("[x%d]",x);
					}
					if (y>=0) {
						yString = String.format("[y+%d]",y);
					} else {
						yString = String.format("[y%d]",y);
					}
					if (z>=0) {
						zString = String.format("[z+%d]",z);
					} else {
						zString = String.format("[z%d]",z);
					}
					System.out.format(".add(volume%s%s%s.multiply(kernel[%d][%d][%d]))",xString,yString,zString, x, y, z);
					//System.out.format(".add(volume%s%s.multiply(kernel[%d][%d]))",xString,yString, x, y);
					if (x == halfWidth && y == halfHeight && z == halfDepth) {
						System.out.format(";%n");
					} else {
						System.out.format("%n");
					}
				}
			}
		}
	}
	
	public static void main(String[] args) {
		
		Unroller.makeUnrollComplex(3,3);
		Unroller.makeUnrollComplex(5,5);
		Unroller.makeUnrollComplex(3,3,3);
		Unroller.makeUnrollComplex(5,5,5);
		
		
	}

}
