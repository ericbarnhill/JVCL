package com.ericbarnhill.jvcl;

public class ConvolverFactory {

    public enum DataType {
        FLOAT, DOUBLE, COMPLEX
    }

    public enum ConvolutionType {
        FDCPU, FTCPU, FDGPU
    }

    private final ConvolutionType convolutionType;

    public ConvolverFactory() {
        this.convolutionType = ConvolutionType.FDCPU;
    }

    @SuppressWarnings("unchecked")
    public Convolver getConvolver(DataType dataType, ConvolutionType convolutionType) {
        switch (dataType) { 
            case FLOAT:
                switch(convolutionType) {
                    case FDCPU:
                        return new ConvolverFloatFDCPU();
                    case FDGPU:
                        return new ConvolverFloatFDGPU();
                    case FTCPU:
                        return new ConvolverFloatFTCPU();
                }
                break;
            case DOUBLE:
                switch(convolutionType) {
                    case FDCPU:
                        return new ConvolverDoubleFDCPU();
                    case FDGPU:
                        return new ConvolverDoubleFDGPU();
                    case FTCPU:
                        return new ConvolverDoubleFTCPU();
                }
                break;
            case COMPLEX:
                switch(convolutionType) {
                    case FDCPU:
                        return new ConvolverComplexFDCPU();
                    case FDGPU:
                        return new ConvolverComplexFDGPU();
                    case FTCPU:
                        return new ConvolverComplexFTCPU();
                }
        }
        throw new RuntimeException("JVCL: Type error in convolver"); 
    }
}

        
    

