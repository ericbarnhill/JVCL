package com.ericbarnhill.jvcl;

import org.apache.commons.math4.complex.Complex;

import com.ericbarnhill.arrayMath.ArrayMath;

public class Boundaries {


	public static double[] finishBoundaries1d(double[] fPad, double[] gg, double[] r,
			int gi, int hgi, int hgie, int ri) {
		int ai;
		for (int i = 0; i < hgie; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i] += fPad[ai]*gg[(gi-1-p)];
				}
			}
		}
		for (int i = ri-hgi-1; i < ri; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i] += fPad[ai]*gg[(gi-1-p)];
				}
			}
		}
		return r;
	}
	
	public static Complex[] finishBoundaries1d(Complex[] fPad, Complex[] gg, Complex[] r,
			int gi, int hgi, int hgie, int ri) {
		int ai;
		for (int i = 0; i < hgi; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i].add(fPad[ai].multiply(gg[(gi-1-p)]));
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i].add(fPad[ai].multiply(gg[(gi-1-p)]));
				}
			}
		}
		return r;
	}
	
	public static double[][] finishBoundaries2d(double[] fPad, double[] gg, double[] r,
			int gi, int gj, int hgi, int hgie, int hgj, int hgje, int ri, int rj) {
		int ai, aj;
		for (int i = 0; i < hgi; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*gg[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*gg[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*gg[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*gg[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		return ArrayMath.devectorise(r, ri);
	}
	

	public static Complex[][] finishBoundaries2d(Complex[] fPad, Complex[] gg, Complex[] r,
			int gi, int gj, int hgi, int hgie, int hgj, int hgje, int ri, int rj) {
		int ai, aj;
		for (int i = 0; i < hgi; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(gg[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(gg[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(gg[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(gg[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		return ArrayMath.devectorise(r, ri);
	}
	

	public static double[][][] finishBoundaries3d(double[] fPad, double[] gg, double[] r,
			int gi, int gj, int gk, int hgi, int hgie, int hgj, int hgje, int hgk, int hgke, 
			int ri, int rj, int rk) {
		boolean isBoundary = true;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					isBoundary = (i > hgi && i < ri-hgie-1 && j > hgj && j < rj-hgje-1 &&
							k > hgk && k < rk - hgke - 1) ? false : true;
					if (!isBoundary) break;
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i + j*ri + k*ri*rj] += fPad[ai + aj*ri + ak*ri*rj]*gg[gi-1-p + (gj-1-q) *gi + (gk-1-s)*gi*gj];
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
		return ArrayMath.devectorise(r, ri, rj);
	}
	

	public static Complex[][][] finishBoundaries3d(Complex[] fPad, Complex[] gg, Complex[] r,
			int gi, int gj, int gk, int hgi, int hgie, int hgj, int hgje, int hgk, int hgke, 
			int ri, int rj, int rk) {
		boolean isBoundary = true;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					isBoundary = (i > hgi && i < ri-hgie-1 && j > hgj && j < rj-hgje-1 &&
							k > hgk && k < rk - hgke - 1) ? false : true;
					if (!isBoundary) break;
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i + j*ri + k*ri*rj].add(fPad[ai + aj*ri + ak*ri*rj].multiply(gg[gi-1-p + (gj-1-q) *gi + (gk-1-s)*gi*gj]));
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
		return ArrayMath.devectorise(r, ri, rj);
	}
	

	
	
}
