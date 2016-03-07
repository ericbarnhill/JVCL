# JVCL

Java Volumetric Convolution Library 0.1 (JVCL) (c) Eric Barnhill 2016 All Rights Reserved.

Note: This package is in alpha until I code the JUnit tests, then it will move to beta.

The Java Volumetric Convolution Library (JVCL) aims to create optimised code design and performance for image processing convolution operations in Java.

User has the choice of finite-difference or FFT-based convolutions, both on the CPU with support for multi-core operations, and the GPU via openCL. The user must have set up OpenCL and installed JogAmp to use OpenCL.

Particular emphasis is on support for Complex valued, three-dimensional medical image volumes and the library uses the Complex object and supporting classes from [Apache commons-math] (https://commons.apache.org/proper/commons-math/), as well as array-wise operations from [Java ArrayMath] (https://github.com/ericbarnhill/java-array-math)

Formal testing is still to come, however initial tests suggested that contrary to the conventional wisdom about the JVM, vectorised unrolled operations are much faster than naive implementations on the CPU. This is all the more true for GPU operations, although the GPU operations, again in initial testing, easily outperform all CPU operations. 

To further support convolution unrolling, an Unroller class is in the package which will create a public unrolled Convolution method to dimensions of your specification, and add it to the Unrolled class.

Javadoc to come.

JVCL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
  
JVCL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with JVCL.  If not, see http://www.gnu.org/licenses/ .
 
This code uses software from the Apache Software Foundation. The Apache Software License can be found at: http://www.apache.org/licenses/LICENSE-2.0.txt .

This code uses libraries from the JogAmp software package, for more information and licenses see https://jogamp.org/ .

